#pragma once

struct ConformanceResult
{
    bool passed = true;
    float epsilon = 0.0f;
    float biggest_difference = 0.0f;
    float node_value = 0.0f;
    float reference_value = 0.0f;
    std::uint32_t index = 0;
    std::size_t tested_samples_count = 0;
};

template<typename Dt>
inline ConformanceResult run_conformance_check(const std::vector<std::byte>& gpu_untyped_result, const std::vector<std::byte>& dnnl_untyped_result, float epsilon, bool print_mismatches)
{
    const auto* gpu_typed_result = reinterpret_cast<const Dt*>(gpu_untyped_result.data());
    const auto* dnnl_typed_result = reinterpret_cast<const Dt*>(dnnl_untyped_result.data());

    // compare results
    ConformanceResult ret;
    ret.epsilon = epsilon;
    for (std::uint32_t i = 0; i < gpu_untyped_result.size() / sizeof(Dt); i++)
    {
        ret.node_value = cast_to_float(gpu_typed_result[i]);
        ret.reference_value = cast_to_float(dnnl_typed_result[i]);

        const auto abs_diff = std::abs(ret.node_value - ret.reference_value);

        if (abs_diff > ret.epsilon || std::isnan(ret.node_value) || std::isnan(ret.reference_value))
        {
            ret.passed = false;

            if (print_mismatches)
            {
                std::cout << std::format("Mismatch, gpu: {}, cpu: {}, at index: {}. Absolute differece: {} \n", ret.node_value, ret.reference_value, i, abs_diff);
            }

        }
        ret.biggest_difference = std::max(ret.biggest_difference, abs_diff);
        ret.tested_samples_count++;
    }
    return ret;
}