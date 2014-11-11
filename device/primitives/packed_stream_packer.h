/*
 * Copyright (c) 2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 *
 *
 *
 *
 *
 *
 *
 */

#pragma once

#include "util.h"

namespace firepony {

// lifted directly from nvbio
// xxxnsubtil: could use some cleanup

template <bool BIG_ENDIAN_T, uint32 SYMBOL_SIZE, typename Symbol, typename InputStream, typename IndexType, typename ValueType>
struct packer {
};

template <bool BIG_ENDIAN_T, uint32 SYMBOL_SIZE, typename Symbol, typename InputStream, typename IndexType>
struct packer<BIG_ENDIAN_T,SYMBOL_SIZE,Symbol,InputStream,IndexType,uint32>
{
    static CUDA_HOST_DEVICE Symbol get_symbol(InputStream stream, const IndexType sym_idx)
    {
        const uint32 SYMBOL_COUNT = 1u << SYMBOL_SIZE;
        const uint32 SYMBOL_MASK  = SYMBOL_COUNT - 1u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const uint64     bit_idx  = uint64(sym_idx) * SYMBOL_SIZE;
        const index_type word_idx = index_type( bit_idx >> 5u );

        if (is_pow2(SYMBOL_SIZE))
        {
            const uint32 word = stream[ word_idx ];
            const uint32 symbol_offset = BIG_ENDIAN_T ? (32u - SYMBOL_SIZE  - uint32(bit_idx & 31u)) : uint32(bit_idx & 31u);
            const uint32 symbol = (word >> symbol_offset) & SYMBOL_MASK;

            return Symbol( symbol );
        }
        else
        {
            const uint32 word1 = stream[ word_idx ];
            const uint32 symbol_offset = uint32(bit_idx & 31u);
            const uint32 symbol1 = (word1 >> symbol_offset) & SYMBOL_MASK;

            // check if we need to read a second word
            const uint32 read_bits = min( 32u - symbol_offset, SYMBOL_SIZE );
            const uint32 rem_bits  = SYMBOL_SIZE - read_bits;
            if (rem_bits)
            {
                const uint32 rem_mask = (1u << rem_bits) - 1u;

                const uint32 word2 = stream[ word_idx+1 ];
                const uint32 symbol2 = word2 & rem_mask;

                return Symbol( symbol1 | (symbol2 << read_bits) );
            }
            else
                return Symbol( symbol1 );
        }
    }

    static CUDA_HOST_DEVICE void set_symbol(InputStream stream, const IndexType sym_idx, Symbol sym)
    {
        const uint32 SYMBOL_COUNT = 1u << SYMBOL_SIZE;
        const uint32 SYMBOL_MASK  = SYMBOL_COUNT - 1u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const uint64     bit_idx  = uint64(sym_idx) * SYMBOL_SIZE;
        const index_type word_idx = index_type( bit_idx >> 5u );

        if (is_pow2(SYMBOL_SIZE))
        {
                  uint32 word          = stream[ word_idx ];
            const uint32 symbol_offset = BIG_ENDIAN_T ? (32u - SYMBOL_SIZE - uint32(bit_idx & 31u)) : uint32(bit_idx & 31u);
            const uint32 symbol        = uint32(sym & SYMBOL_MASK) << symbol_offset;

            // clear all bits
            word &= ~(SYMBOL_MASK << symbol_offset);

            // set bits
            stream[ word_idx ] = word | symbol;
        }
        else
        {
                  uint32 word1         = stream[ word_idx ];
            const uint32 symbol_offset = uint32(bit_idx & 31u);
            const uint32 symbol1       = uint32(sym & SYMBOL_MASK) << symbol_offset;

            // clear all bits
            word1 &= ~(SYMBOL_MASK << symbol_offset);

            // set bits
            stream[ word_idx ] = word1 | symbol1;

            // check if we need to write a second word
            const uint32 read_bits = min( 32u - symbol_offset, SYMBOL_SIZE );
            const uint32 rem_bits  = SYMBOL_SIZE - read_bits;
            if (rem_bits)
            {
                const uint32 rem_mask = (1u << rem_bits) - 1u;

                      uint32 word2   = stream[ word_idx+1 ];
                const uint32 symbol2 = uint32(sym & SYMBOL_MASK) >> read_bits;

                // clear all bits
                word2 &= ~rem_mask;

                // set bits
                stream[ word_idx+1 ] = word2 | symbol2;
            }
        }
    }
};

template <bool BIG_ENDIAN_T, uint32 SYMBOL_SIZE, typename Symbol, typename InputStream, typename IndexType>
struct packer<BIG_ENDIAN_T,SYMBOL_SIZE,Symbol,InputStream,IndexType,uint64>
{
    static CUDA_HOST_DEVICE Symbol get_symbol(InputStream stream, const IndexType sym_idx)
    {
        const uint32 SYMBOL_COUNT = 1u << SYMBOL_SIZE;
        const uint32 SYMBOL_MASK  = SYMBOL_COUNT - 1u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const uint64     bit_idx  = uint64(sym_idx) * SYMBOL_SIZE;
        const index_type word_idx = index_type( bit_idx >> 6u );

        if (is_pow2(SYMBOL_SIZE))
        {
            const uint64 word          = stream[ word_idx ];
            const uint32 symbol_offset = BIG_ENDIAN_T ? (64u - SYMBOL_SIZE  - uint32(bit_idx & 63u)) : uint32(bit_idx & 63u);
            const uint32 symbol        = uint32((word >> symbol_offset) & SYMBOL_MASK);

            return Symbol( symbol );
        }
        else
        {
            const uint64 word1         = stream[ word_idx ];
            const uint32 symbol_offset = uint32(bit_idx & 63u);
            const uint32 symbol1       = uint32((word1 >> symbol_offset) & SYMBOL_MASK);

            // check if we need to read a second word
            const uint32 read_bits = min( 64u - symbol_offset, SYMBOL_SIZE );
            const uint32 rem_bits  = SYMBOL_SIZE - read_bits;
            if (rem_bits)
            {
                const uint64 rem_mask = (uint64(1u) << rem_bits) - 1u;

                const uint64 word2    = stream[ word_idx+1 ];
                const uint32 symbol2  = uint32(word2 & rem_mask);

                return Symbol( symbol1 | (symbol2 << read_bits) );
            }
            else
                return Symbol( symbol1 );
        }
    }

    static CUDA_HOST_DEVICE void set_symbol(InputStream stream, const IndexType sym_idx, Symbol sym)
    {
        const uint32 SYMBOL_COUNT = 1u << SYMBOL_SIZE;
        const uint32 SYMBOL_MASK  = SYMBOL_COUNT - 1u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const uint64     bit_idx  = uint64(sym_idx) * SYMBOL_SIZE;
        const index_type word_idx = index_type( bit_idx >> 6u );

        if (is_pow2(SYMBOL_SIZE))
        {
                  uint64 word = stream[ word_idx ];
            const uint32 symbol_offset = BIG_ENDIAN_T ? (64u - SYMBOL_SIZE - uint32(bit_idx & 63u)) : uint32(bit_idx & 63u);
            const uint64 symbol = uint64(sym & SYMBOL_MASK) << symbol_offset;

            // clear all bits
            word &= ~(uint64(SYMBOL_MASK) << symbol_offset);

            // set bits
            stream[ word_idx ] = word | symbol;
        }
        else
        {
                  uint64 word1 = stream[ word_idx ];
            const uint32 symbol_offset = uint32(bit_idx & 63);
            const uint64 symbol1 = uint64(sym & SYMBOL_MASK) << symbol_offset;

            // clear all bits
            word1 &= ~(uint64(SYMBOL_MASK) << symbol_offset);

            // set bits
            stream[ word_idx ] = word1 | symbol1;

            // check if we need to write a second word
            const uint32 read_bits = min( 64u - symbol_offset, SYMBOL_SIZE );
            const uint32 rem_bits  = SYMBOL_SIZE - read_bits;
            if (rem_bits)
            {
                const uint64 rem_mask = (uint64(1u) << rem_bits) - 1u;

                      uint64 word2   = stream[ word_idx+1 ];
                const uint64 symbol2 = uint64(sym & SYMBOL_MASK) >> read_bits;

                // clear all bits
                word2 &= ~rem_mask;

                // set bits
                stream[ word_idx+1 ] = word2 | symbol2;
            }
        }
    }
};

#if 0
template <bool BIG_ENDIAN_T, uint32 SYMBOL_SIZE, typename Symbol, typename InputStream, typename IndexType>
struct packer<BIG_ENDIAN_T,SYMBOL_SIZE,Symbol,InputStream,IndexType,uint8>
{
    static CUDA_HOST_DEVICE Symbol get_symbol(InputStream stream, const IndexType sym_idx)
    {
        const uint8 SYMBOL_COUNT = uint8(1u) << SYMBOL_SIZE;
        const uint8 SYMBOL_MASK  = SYMBOL_COUNT - uint8(1u);

        typedef typename unsigned_type<IndexType>::type index_type;

        const uint64     bit_idx  = uint64(sym_idx) * SYMBOL_SIZE;
        const index_type word_idx = index_type( bit_idx >> 3u );

        if (is_pow2(SYMBOL_SIZE))
        {
            const uint8 word = stream[ word_idx ];
            const uint8 symbol_offset = BIG_ENDIAN_T ? (8u - SYMBOL_SIZE - uint8(bit_idx & 7u)) : uint8(bit_idx & 7u);
            const uint8 symbol = (word >> symbol_offset) & SYMBOL_MASK;

            return Symbol( symbol );
        }
        else
        {
            const uint8 word1 = stream[ word_idx ];
            const uint8 symbol_offset = uint8(bit_idx & 7u);
            const uint8 symbol1 = (word1 >> symbol_offset) & SYMBOL_MASK;

            // check if we need to read a second word
            const uint32 read_bits = min( 8u - symbol_offset, SYMBOL_SIZE );
            const uint32 rem_bits  = SYMBOL_SIZE - read_bits;
            if (rem_bits)
            {
                const uint8 rem_mask = uint8((1u << rem_bits) - 1u);

                const uint8 word2 = stream[ word_idx+1 ];
                const uint8 symbol2 = word2 & rem_mask;

                return Symbol( symbol1 | (symbol2 << read_bits) );
            }
            else
                return Symbol( symbol1 );
        }
    }

    static CUDA_HOST_DEVICE void set_symbol(InputStream stream, const IndexType sym_idx, Symbol sym)
    {
        const uint8 SYMBOL_COUNT = uint8(1u) << SYMBOL_SIZE;
        const uint8 SYMBOL_MASK  = SYMBOL_COUNT - uint8(1u);

        typedef typename unsigned_type<IndexType>::type index_type;

        const uint64     bit_idx  = uint64(sym_idx) * SYMBOL_SIZE;
        const index_type word_idx = index_type( bit_idx >> 3u );

        if (is_pow2(SYMBOL_SIZE))
        {
                  uint8 word = stream[ word_idx ];
            const uint8 symbol_offset = BIG_ENDIAN_T ? (8u - SYMBOL_SIZE - uint8(bit_idx & 7u)) : uint8(bit_idx & 7u);
            const uint8 symbol = uint32(sym & SYMBOL_MASK) << symbol_offset;

            // clear all bits
            word &= ~(SYMBOL_MASK << symbol_offset);

            // set bits
            stream[ word_idx ] = word | symbol;
        }
        else
        {
                  uint8 word1 = stream[ word_idx ];
            const uint8 symbol_offset = uint8(bit_idx & 7u);
            const uint8 symbol1 = uint8(sym & SYMBOL_MASK) << symbol_offset;

            // clear all bits
            word1 &= ~(SYMBOL_MASK << symbol_offset);

            // set bits
            stream[ word_idx ] = word1 | symbol1;

            // check if we need to write a second word
            const uint32 read_bits = min( 8u - symbol_offset, SYMBOL_SIZE );
            const uint32 rem_bits  = SYMBOL_SIZE - read_bits;
            if (rem_bits)
            {
                      uint8 word2   = stream[ word_idx+1 ];
                const uint8 symbol2 = uint32(sym & SYMBOL_MASK) >> read_bits;

                const uint8 rem_mask = uint8((1u << rem_bits) - 1u);

                // clear all bits
                word2 &= ~rem_mask;

                // set bits
                stream[ word_idx+1 ] = word2 | symbol2;
            }
        }
    }
};
#endif

template <bool BIG_ENDIAN_T, typename Symbol, typename InputStream, typename IndexType>
struct packer<BIG_ENDIAN_T,2u,Symbol,InputStream,IndexType,uint32>
{
    static CUDA_HOST_DEVICE Symbol get_symbol(InputStream stream, const IndexType sym_idx)
    {
        const uint32 SYMBOL_MASK = 3u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 4u;

        const uint32 word = stream[ word_idx ];
        const uint32 symbol_offset = BIG_ENDIAN_T ? (30u - (uint32(sym_idx & 15u) << 1)) : uint32((sym_idx & 15u) << 1);
        const uint32 symbol = (word >> symbol_offset) & SYMBOL_MASK;

        return Symbol( symbol );
    }

    static CUDA_HOST_DEVICE void set_symbol(InputStream stream, const IndexType sym_idx, Symbol sym)
    {
        const uint32 SYMBOL_MASK = 3u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 4u;

              uint32 word = stream[ word_idx ];
        const uint32 symbol_offset = BIG_ENDIAN_T ? (30u - (uint32(sym_idx & 15u) << 1)) : uint32((sym_idx & 15u) << 1);
        const uint32 symbol = uint32(sym & SYMBOL_MASK) << symbol_offset;

        // clear all bits
        word &= ~(SYMBOL_MASK << symbol_offset);

        // set bits
        stream[ word_idx ] = word | symbol;
    }
};
template <bool BIG_ENDIAN_T, typename Symbol, typename InputStream, typename IndexType>
struct packer<BIG_ENDIAN_T,4u,Symbol,InputStream,IndexType,uint32>
{
    static CUDA_HOST_DEVICE Symbol get_symbol(InputStream stream, const IndexType sym_idx)
    {
        const uint32 SYMBOL_MASK = 15u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 3u;

        const uint32 word = stream[ word_idx ];
        const uint32 symbol_offset = BIG_ENDIAN_T ? (28u - (uint32(sym_idx & 7u) << 2)) : uint32((sym_idx & 7u) << 2);
        const uint32 symbol = (word >> symbol_offset) & SYMBOL_MASK;

        return Symbol( symbol );
    }

    static CUDA_HOST_DEVICE void set_symbol(InputStream stream, const IndexType sym_idx, Symbol sym)
    {
        const uint32 SYMBOL_MASK = 15u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 3u;

              uint32 word = stream[ word_idx ];
        const uint32 symbol_offset = BIG_ENDIAN_T ? (28u - (uint32(sym_idx & 7u) << 2)) : uint32((sym_idx & 7u) << 2);
        const uint32 symbol = uint32(sym & SYMBOL_MASK) << symbol_offset;

        // clear all bits
        word &= ~(SYMBOL_MASK << symbol_offset);

        // set bits
        stream[ word_idx ] = word | symbol;
    }
};

#if 0
template <bool BIG_ENDIAN_T, typename Symbol, typename InputStream, typename IndexType>
struct packer<BIG_ENDIAN_T,2u,Symbol,InputStream,IndexType,uint4>
{
    static CUDA_HOST_DEVICE Symbol get_symbol(InputStream stream, const IndexType sym_idx)
    {
        const uint32 SYMBOL_MASK = 3u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 6u;

        const uint4  word = stream[ word_idx ];
        const uint32 symbol_comp   = (sym_idx & 63u) >> 4u;
        const uint32 symbol_offset = BIG_ENDIAN_T ? (30u - (uint32(sym_idx & 15u) << 1)) : uint32((sym_idx & 15u) << 1);
        const uint32 symbol = (comp( word, symbol_comp ) >> symbol_offset) & SYMBOL_MASK;

        return Symbol( symbol );
    }

    static CUDA_HOST_DEVICE void set_symbol(InputStream stream, const IndexType sym_idx, Symbol sym)
    {
        const uint32 SYMBOL_MASK = 3u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 6u;

              uint4  word = stream[ word_idx ];
        const uint32 symbol_comp   = (sym_idx & 63u) >> 4u;
        const uint32 symbol_offset = BIG_ENDIAN_T ? (30u - (uint32(sym_idx & 15u) << 1)) : uint32((sym_idx & 15u) << 1);
        const uint32 symbol = uint32(sym & SYMBOL_MASK) << symbol_offset;

        // clear all bits
        select( word, symbol_comp ) &= ~(SYMBOL_MASK << symbol_offset);
        select( word, symbol_comp ) |= symbol;

        // set bits
        stream[ word_idx ] = word;
    }
};
template <bool BIG_ENDIAN_T, typename Symbol, typename InputStream, typename IndexType>
struct packer<BIG_ENDIAN_T,4u,Symbol,InputStream,IndexType,uint4>
{
    static CUDA_HOST_DEVICE Symbol get_symbol(InputStream stream, const IndexType sym_idx)
    {
        const uint32 SYMBOL_MASK = 15u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 5u;

        const uint4 word = stream[ word_idx ];
        const uint32 symbol_comp   = (sym_idx & 31u) >> 3u;
        const uint32 symbol_offset = BIG_ENDIAN_T ? (28u - (uint32(sym_idx & 7u) << 2)) : uint32((sym_idx & 7u) << 2);
        const uint32 symbol = (comp( word, symbol_comp ) >> symbol_offset) & SYMBOL_MASK;

        return Symbol( symbol );
    }

    static CUDA_HOST_DEVICE void set_symbol(InputStream stream, const IndexType sym_idx, Symbol sym)
    {
        const uint32 SYMBOL_MASK = 15u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 5u;

              uint4  word = stream[ word_idx ];
        const uint32 symbol_comp   = (sym_idx & 31u) >> 3u;
        const uint32 symbol_offset = BIG_ENDIAN_T ? (28u - (uint32(sym_idx & 7u) << 2)) : uint32((sym_idx & 7u) << 2);
        const uint32 symbol = uint32(sym & SYMBOL_MASK) << symbol_offset;

        // clear all bits
        select( word, symbol_comp ) &= ~(SYMBOL_MASK << symbol_offset);
        select( word, symbol_comp ) |= symbol;

        // set bits
        stream[ word_idx ] = word;
    }
};
#endif

template <bool BIG_ENDIAN_T, typename Symbol, typename InputStream, typename IndexType>
struct packer<BIG_ENDIAN_T,2u,Symbol,InputStream,IndexType,uint64>
{
    static CUDA_HOST_DEVICE Symbol get_symbol(InputStream stream, const IndexType sym_idx)
    {
        const uint32 SYMBOL_MASK = 3u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 5u;

        const uint64 word = stream[ word_idx ];
        const uint32 symbol_offset = BIG_ENDIAN_T ? (62u - (uint32(sym_idx & 31u) << 1)) : uint32((sym_idx & 31u) << 1);
        const uint64 symbol = (word >> symbol_offset) & SYMBOL_MASK;

        return Symbol( symbol );
    }

    static CUDA_HOST_DEVICE void set_symbol(InputStream stream, const IndexType sym_idx, Symbol sym)
    {
        const uint32 SYMBOL_MASK = 3u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 5u;

              uint64 word = stream[ word_idx ];
        const uint32 symbol_offset = BIG_ENDIAN_T ? (62u - (uint32(sym_idx & 31u) << 1)) : uint32((sym_idx & 31u) << 1);
        const uint64 symbol = uint64(sym & SYMBOL_MASK) << symbol_offset;

        // clear all bits
        word &= ~(uint64(SYMBOL_MASK) << symbol_offset);

        // set bits
        stream[ word_idx ] = word | symbol;
    }
};
template <bool BIG_ENDIAN_T, typename Symbol, typename InputStream, typename IndexType>
struct packer<BIG_ENDIAN_T,4u,Symbol,InputStream,IndexType,uint64>
{
    static CUDA_HOST_DEVICE Symbol get_symbol(InputStream stream, const IndexType sym_idx)
    {
        const uint32 SYMBOL_MASK = 15u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 5u;

        const uint64 word = stream[ word_idx ];
        const uint32 symbol_offset = BIG_ENDIAN_T ? (60u - (uint32(sym_idx & 15u) << 2)) : uint32((sym_idx & 15u) << 2);
        const uint64 symbol = (word >> symbol_offset) & SYMBOL_MASK;

        return Symbol( symbol );
    }

    static CUDA_HOST_DEVICE void set_symbol(InputStream stream, const IndexType sym_idx, Symbol sym)
    {
        const uint32 SYMBOL_MASK = 15u;

        typedef typename unsigned_type<IndexType>::type index_type;

        const index_type word_idx = sym_idx >> 5u;

              uint64 word = stream[ word_idx ];
        const uint32 symbol_offset = BIG_ENDIAN_T ? (60u - (uint32(sym_idx & 15u) << 2)) : uint32((sym_idx & 15u) << 2);
        const uint64 symbol = uint32(sym & SYMBOL_MASK) << symbol_offset;

        // clear all bits
        word &= ~(SYMBOL_MASK << symbol_offset);

        // set bits
        stream[ word_idx ] = word | symbol;
    }
};

} // namespace firepony
