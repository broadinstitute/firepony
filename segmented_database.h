/*
 * Firepony
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the copyright holders nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "types.h"

namespace firepony {

// represents a resident set for the segmented database
struct resident_segment_map
{
    persistent_allocation<host, bool> set;

    resident_segment_map()
        : set()
    { }

    resident_segment_map(uint16 size)
        : set()
    {
        resize(size);
        clear();
    }

    LIFT_HOST_DEVICE uint32 size(void) const
    {
        return set.size();
    }

    LIFT_HOST_DEVICE bool is_resident(uint16 segment) const
    {
        if (segment >= set.size())
        {
            return false;
        }

        return set[segment];
    }

    // a resident segment set can never shrink
    // this is because any sequences that have been referenced before
    // will be present in the host-side memory database
    void resize(uint16 num_sequences)
    {
        if (set.size() < num_sequences)
        {
            size_t old_size = set.size();
            set.resize(num_sequences);
            for(size_t i = old_size; i < num_sequences; i++)
            {
                set[i] = false;
            }
        }
    }

    void mark_resident(uint16 segment)
    {
        resize(segment + 1);
        set[segment] = true;
    }

    void mark_evicted(uint16 segment)
    {
        resize(segment + 1);
        set[segment] = false;
    }

    void clear(void)
    {
        for(uint32 i = 0; i < set.size(); i++)
        {
            mark_evicted(i);
        }
    }
};

// a generic database segmented by chromosome, used for both reference and variant data
// storage type is meant to be a structure that holds the data for a given chromosome
// it must contain a const_view type and implement the appropriate conversion operator
template <target_system system,
          template <target_system _unused> class chromosome_storage>
struct segmented_database_storage
{
    // per-chromosome data
    persistent_allocation<system, chromosome_storage<system>> storage;
    resident_segment_map storage_map;

    segmented_database_storage()
        : storage(), storage_map()
    { }

    // check if a given sequence is resident in the database
    LIFT_HOST_DEVICE bool is_resident(uint16 id) const
    {
        return storage_map.is_resident(id);
    }

    // look up a sequence in the database and return a reference
    LIFT_HOST_DEVICE const chromosome_storage<system>& get_sequence(uint16 id) const
    {
        return storage[id];
    }

    LIFT_HOST_DEVICE chromosome_storage<system>& get_sequence(uint16 id)
    {
        return storage[id];
    }

private:
    // evict chromosome at index i
    void evict(uint16 i)
    {
        if (storage_map.is_resident(i))
        {
            storage.peek(i).free();
            storage.poke(i, chromosome_storage<system>());
            storage_map.mark_evicted(i);
        }
    }

    // make chromosome i resident
    void download(const segmented_database_storage<host, chromosome_storage>& db,
                  uint16 i)
    {
        if (!storage_map.is_resident(i))
        {
            // copy data
            // the song and dance below is required since storage.peek(i) returns a copy of the object
            // we need to grab the pointer, modify it, then poke it back into the array
            auto container = storage.peek(i);
            container.copy(db.storage[i]);
            storage.poke(i, container);

            storage_map.mark_resident(i);
        }
    }

    // initialize a range [start, end[ of storage pointers
    void initialize_range(size_t start, size_t end)
    {
        for(size_t i = start; i < end; i++)
        {
            if (system == host)
            {
                // use placement new to initialize the objects
                // this is required to properly initialize any vtables for host storage objects
                auto *ptr = storage.data() + i;
                new (ptr) chromosome_storage<system>();
            } else {
                // vtables are not valid on the device; initialize by assignment
                storage.poke(i, chromosome_storage<system>());
            }
        }
    }

    void resize(size_t new_size)
    {
        if (storage.size() < new_size)
        {
            size_t old_size = storage.size();

            storage.resize(new_size);
            storage_map.resize(new_size);
            initialize_range(old_size, storage.size());
        }
    }

public:
    // creates an entry for a given sequence ID
    // returns nullptr if the given ID already exists in the database
    chromosome_storage<system> *new_entry(uint16 id)
    {
        static_assert(system == host, "segmented_database::new_entry can not be called for device storage");

        // make sure we have enough slots
        resize(size_t(id + 1));

        if (storage_map.is_resident(id))
        {
            return nullptr;
        }

        // the code below is valid since new_entry is only called for host databases
        storage[id] = chromosome_storage<system>();
        storage_map.mark_resident(id);
        return &storage[id];
    }

    // returns a resident segment map of the right size for the current database with all entries marked non-resident
    resident_segment_map empty_segment_map(void) const
    {
        return resident_segment_map(storage.size());
    }

    // make a set of chromosomes resident, evict any not marked as resident in the set
    void update_resident_set(const segmented_database_storage<host, chromosome_storage>& db,
                             const resident_segment_map& target_resident_set)
    {
        // note: the resident segment map can in some cases be larger than the current database
        // this can happen if a chromosome referenced in the alignment input is present in the reference
        // but not the dbsnp
        assert(target_resident_set.size() >= db.storage.size());

        // make sure we have enough slots in the database
        resize(db.storage.size());

        for(uint32 i = 0; i < db.storage.size(); i++)
        {
            if (target_resident_set.is_resident(i))
            {
                download(db, i);
            } else {
                evict(i);
            }
        }
    }

    LIFT_HOST_DEVICE uint32 size(void) const
    {
        return storage.size();
    }
};

} // namespace firepony
