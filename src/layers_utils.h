#pragma once
#include <cassert>
#include <cstdint>
enum class DataType
{
    eFp32 = 0,
    eFp16 = 1,
    eCount
};

inline std::uint8_t get_data_type_bytes_width(DataType dt)
{
    switch (dt)
    {
    case DataType::eFp32: return sizeof(float);
    case DataType::eFp16: return sizeof(std::uint16_t);
    default:
        assert(false && "Unknown data type.");
    }
    return 0;
}

enum class DataLayout
{
    eNCHW = 0,
    eNHWC = 1,
    eCount
};

template<typename T>
inline constexpr T round_up_next_multiple(T N, T M) 
{
    return ((N + M - 1) / M) * M;
}