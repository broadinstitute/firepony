/*
 * Copyright (c) 2012-14, NVIDIA CORPORATION.  All rights reserved.
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

#include "../bqsr_types.h"

namespace from_nvbio {

///@addtogroup Strings
///@{

///\defgroup AlphabetsModule Alphabets
///\par
/// This module provides various operators to work with the following alphabets:
///\par
/// <table>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">DNA</td>
/// <td style="vertical-align:text-top;">4-letter DNA alphabet</td>
/// <td style="vertical-align:text-top;">A,C,G,T</td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">DNA_N</td>
/// <td style="vertical-align:text-top;">5-letter DNA + N alphabet</td>
/// <td style="vertical-align:text-top;">A,C,G,T,N</td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">DNA_IUPAC</td>
/// <td style="vertical-align:text-top;">16-letter DNA IUPAC alphabet</td>
/// <td style="vertical-align:text-top;">=,A,C,M,G,R,S,V,T,W,Y,H,K,D,B,N</td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">PROTEIN</td>
/// <td style="vertical-align:text-top;">24-letter Protein alphabet</td>
/// <td style="vertical-align:text-top;">A,C,D,E,F,G,H,I,K,L,M,N,O,P,Q,R,S,T,V,W,Y,B,Z,X</td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">RNA</td>
/// <td style="vertical-align:text-top;">4-letter RNA alphabet</td>
/// <td style="vertical-align:text-top;">A,C,G,U</td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">RNA_N</td>
/// <td style="vertical-align:text-top;">5-letter RNA + N alphabet</td>
/// <td style="vertical-align:text-top;">A,C,G,U,N</td></tr>
/// </table>
///
///@{

///
/// The supported sequence alphabet types
///
enum Alphabet
{
    DNA       = 0u,           ///< 4-letter DNA alphabet        { A,C,G,T }
    DNA_N     = 1u,           ///< 5-letter DNA + N alphabet    { A,C,G,T,N }
    DNA_IUPAC = 2u,           ///< 16-letter DNA IUPAC alphabet { =,A,C,M,G,R,S,V,T,W,Y,H,K,D,B,N }
    PROTEIN   = 3u,           ///< 24-letter Protein alphabet   { A,C,D,E,F,G,H,I,K,L,M,N,O,P,Q,R,S,T,V,W,Y,B,Z,X }
    RNA       = 4u,           ///< 4-letter RNA alphabet        { A,C,G,U }
    RNA_N     = 5u,           ///< 5-letter RNA + N alphabet    { A,C,G,U,N }
};

/// A traits class for Alphabet
///
template <Alphabet ALPHABET> struct AlphabetTraits {};

/// A traits class for DNA Alphabet
///
template <> struct AlphabetTraits<DNA>
{
    static const uint32 SYMBOL_SIZE  = 2;
    static const uint32 SYMBOL_COUNT = 4;

    enum {
        A = 0,
        C = 1,
        G = 2,
        T = 3,
    };
};
/// A traits class for DNA_N Alphabet
///
template <> struct AlphabetTraits<DNA_N>
{
    static const uint32 SYMBOL_SIZE  = 4;
    static const uint32 SYMBOL_COUNT = 5;

    enum {
        A = 0,
        C = 1,
        G = 2,
        T = 3,
        N = 4,
    };
};
/// A traits class for DNA_IUPAC Alphabet
///
template <> struct AlphabetTraits<DNA_IUPAC>
{
    static const uint32 SYMBOL_SIZE  = 4;
    static const uint32 SYMBOL_COUNT = 16;

    enum {
        EQUAL = 0,
        A     = 1,
        C     = 2,
        M     = 3,
        G     = 4,
        R     = 5,
        S     = 6,
        V     = 7,
        T     = 8,
        W     = 9,
        Y     = 10,
        H     = 11,
        K     = 12,
        D     = 13,
        B     = 14,
        N     = 15,
    };
};
/// A traits class for Protein Alphabet
///
template <> struct AlphabetTraits<PROTEIN>
{
    static const uint32 SYMBOL_SIZE  = 8;
    static const uint32 SYMBOL_COUNT = 24;

    enum {
        A = 0,
        C = 1,
        D = 2,
        E = 3,
        F = 4,
        G = 5,
        H = 6,
        I = 7,
        K = 8,
        L = 9,
        M = 10,
        N = 11,
        O = 12,
        P = 13,
        Q = 14,
        R = 15,
        S = 16,
        T = 17,
        V = 18,
        W = 19,
        Y = 20,
        B = 21,
        Z = 22,
        X = 23,
    };
};
/// A traits class for DNA Alphabet
///
template <> struct AlphabetTraits<RNA>
{
    static const uint32 SYMBOL_SIZE  = 2;
    static const uint32 SYMBOL_COUNT = 4;

    enum {
        A = 0,
        C = 1,
        G = 2,
        U = 3,
    };
};
/// A traits class for DNA_N Alphabet
///
template <> struct AlphabetTraits<RNA_N>
{
    static const uint32 SYMBOL_SIZE  = 4;
    static const uint32 SYMBOL_COUNT = 5;

    enum {
         A = 0,
         C = 1,
         G = 2,
         U = 3,
         N = 4,
    };
};

/// return the number of bits per symbol for a given alphabet
///
inline CUDA_HOST_DEVICE
uint32 bits_per_symbol(const Alphabet alphabet)
{
    return alphabet == DNA       ? 2 :
           alphabet == DNA_N     ? 4 :
           alphabet == DNA_IUPAC ? 4 :
           alphabet == PROTEIN   ? 8 :
           alphabet == RNA       ? 2 :
           alphabet == RNA_N     ? 4 :
           8u;
}

/// convert a given symbol to its ASCII character
///
template <Alphabet ALPHABET>
inline CUDA_HOST_DEVICE char to_char(const uint8 c);

/// convert a given symbol to its ASCII character
///
template <Alphabet ALPHABET>
inline CUDA_HOST_DEVICE uint8 from_char(const char c);

/// convert from the given alphabet to an ASCII string
///
template <Alphabet ALPHABET, typename SymbolIterator>
inline CUDA_HOST_DEVICE void to_string(
    const SymbolIterator begin,
    const uint32         n,
    char*                string);

/// convert from the given alphabet to an ASCII string
///
template <Alphabet ALPHABET, typename SymbolIterator>
inline CUDA_HOST_DEVICE void to_string(
    const SymbolIterator begin,
    const SymbolIterator end,
    char*                string);

/// convert from an ASCII string to the given alphabet
///
template <Alphabet ALPHABET, typename SymbolIterator>
inline CUDA_HOST_DEVICE void from_string(
    const char*             begin,
    const char*             end,
    SymbolIterator          symbols);

/// convert from an ASCII string to the given alphabet
///
template <Alphabet ALPHABET, typename SymbolIterator>
inline CUDA_HOST_DEVICE void from_string(
    const char*             begin,
    SymbolIterator          symbols);

/// conversion functor from a given alphabet to ASCII char
///
template <Alphabet ALPHABET>
struct to_char_functor
{
    typedef uint8 argument_type;
    typedef char  result_type;

    /// functor operator
    ///
    inline CUDA_HOST_DEVICE char operator() (const uint8 c) const { return to_char<ALPHABET>( c ); }
};

/// conversion functor from a given alphabet to ASCII char
///
template <Alphabet ALPHABET>
struct from_char_functor
{
    typedef char  argument_type;
    typedef uint8 result_type;

    /// functor operator
    ///
    inline CUDA_HOST_DEVICE uint8 operator() (const char c) const { return from_char<ALPHABET>( c ); }
};

///@} AlphabetsModule
///@} Strings

} // namespace from_nvbio

#include "alphabet_inl.h"
