#pragma once
struct TensorShape
{
    std::uint32_t n = 0;
    std::uint32_t c = 0;
    std::uint32_t d = 0; // for 5d tensors
    std::uint32_t h = 0;
    std::uint32_t w = 0;

    TensorShape() = default;

    TensorShape(std::uint32_t n, std::uint32_t c, std::uint32_t h, std::uint32_t w)
        : n(n), c(c), h(h), w(w)
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

    inline std::size_t get_elements_count(std::size_t h_align = 1ull, std::size_t w_align = 1ull) const
    {
        if (get_dims_count() == 0)
        {
            return 0;
        }

        std::size_t size_n = n ? n : 1;
        std::size_t size_c = c ? c : 1;
        std::size_t size_d = d ? d : 1;
        std::size_t size_h = h ? h : 1;
        std::size_t size_w = w ? w : 1;

        std::size_t acc = 1;
        acc *= size_n;
        acc *= size_c;
        acc *= size_d;
        acc *= size_h;
        acc *= size_w;
        return acc;
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
