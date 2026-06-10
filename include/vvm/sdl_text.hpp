#pragma once

#include <SDL3/SDL.h>

namespace vvm {

bool init_text_subsystem();
void shutdown_text_subsystem();

void draw_text(SDL_Renderer* renderer, int point_size, const char* text, float x, float y,
               SDL_Color color);

bool measure_text(int point_size, const char* text, int* width, int* height);

} // namespace vvm
