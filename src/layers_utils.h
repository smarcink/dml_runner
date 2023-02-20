#pragma once
#include <span>
#include <string>
#include <cassert>
#include <cstdint>
#include <istream>
#include <vector>


struct TensorShape
{
    std::uint32_t n = 0;
    std::uint32_t c = 0;
    std::uint32_t d = 0; // for 5d tensors
    std::uint32_t h = 0;
    std::uint32_t w = 0;

    TensorShape() = default;
    TensorShape(std::span<std::uint32_t> in_v)
    {
        assert((in_v.size() == 2 || in_v.size() == 4 || in_v.size() == 5) && "Not supported shape!");
        std::int32_t current_idx = static_cast<std::int32_t>(in_v.size()) - 1;
        w = in_v[current_idx--];
        h = in_v[current_idx--];
        if (in_v.size() == 5)
        {
            d = in_v[current_idx--];
        }
        if (in_v.size() > 2)
        {
            c = in_v[current_idx--];
            n = in_v[current_idx--];
        }
        assert(current_idx == -1 && "Current idex should be equal -1 (parsed all dimensions).");
    }

    //friend std::istream& operator>>(std::istream& input, TensorShape& ts) {
    //    std::vector<std::uint32_t> data;
    //    constexpr const auto buffer_size = 128;
    //    std::string line(buffer_size, ' ');
    //    while(input.getline(line.data(), buffer_size, ','))
    //    {
    //        data.push_back(std::stoi(line));
    //    }
    //    ts = TensorShape(data);
    //    return input;
    //}
};

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