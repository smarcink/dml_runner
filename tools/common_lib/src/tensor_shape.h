#pragma once
struct TensorShape
{
    std::uint32_t n = 0;
    std::uint32_t c = 0;
    std::uint32_t d = 0; // for 5d tensors
    std::uint32_t h = 0;
    std::uint32_t w = 0;

    TensorShape() = default;

    TensorShape(std::uint32_t n, std::uint32_t h, std::uint32_t w)
        : n(n), h(h), w(w)
    {
    }
    TensorShape(std::uint32_t n, std::uint32_t c, std::uint32_t h, std::uint32_t w)
        : n(n), c(c), h(h), w(w)
    {
    }
    TensorShape(std::uint32_t n, std::uint32_t c, std::uint32_t d, std::uint32_t h, std::uint32_t w)
        : n(n), c(c), d(d), h(h), w(w)
    {
    }
    TensorShape(std::span<std::uint32_t> in_v)
    {
        assert(!(in_v.size() < 3 || in_v.size() > 5) && "Not supported shape!");
        std::int32_t current_idx = static_cast<std::int32_t>(in_v.size()) - 1;
        if (in_v.size() > 3)
        {
            w = in_v[current_idx--];
        }
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

    inline std::uint8_t get_dims_count() const
    {
        std::uint8_t ret = 0;
        if (n) ret++;
        if (c) ret++;
        if (d) ret++;
        if (h) ret++;
        if (w) ret++;

        return ret;
    }
};

inline bool lexical_cast(const std::string& input, TensorShape& ts)
{
    std::vector<std::uint32_t> data;
    constexpr const auto buffer_size = 128;
    std::string line(buffer_size, ' ');
    std::stringstream stream;
    stream << input;
    while (stream.getline(line.data(), buffer_size, ','))
    {
        data.push_back(std::stoi(line));
    }
    ts = TensorShape(data);
    return true;
}