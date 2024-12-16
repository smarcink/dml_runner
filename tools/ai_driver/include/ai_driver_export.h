#ifndef AI_DRIVER_EXPORT_H
#define AI_DRIVER_EXPORT_H

#ifdef ai_driver_EXPORTS
#define AI_DRIVER_API __declspec(dllexport)
#else
#define AI_DRIVER_API __declspec(dllimport)
#endif

#endif  // AI_DRIVER_EXPORT_H