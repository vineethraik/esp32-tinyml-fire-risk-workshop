#pragma once

#include <cstddef>
#include <cstdint>

// Minimal Arduino stand-ins for testing the inference library on a PC.
struct __FlashStringHelper {};
#define F(text) reinterpret_cast<const __FlashStringHelper *>(text)

template <typename T>
T max(T first, T second) {
  return first > second ? first : second;
}

inline uint32_t micros() {
  static uint32_t tick = 0;
  return ++tick;
}
