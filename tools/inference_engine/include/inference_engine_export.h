#ifndef INFERENCE_ENGINE_EXPORT_H
#define INFERENCE_ENGINE_EXPORT_H

#ifdef inference_engine_EXPORTS
#define INFERENCE_ENGINE_API __declspec(dllexport)
#else
#define INFERENCE_ENGINE_API __declspec(dllimport)
#endif

#endif  // INFERENCE_ENGINE_EXPORT_H