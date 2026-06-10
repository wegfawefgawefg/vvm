#include "vvm/sdl_text.hpp"

#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace vvm {

namespace {

constexpr int kDefaultFontBaseSize = 10;
constexpr int kDefaultFontCharsCount = 224;
constexpr int kDefaultFontAtlasWidth = 128;
constexpr int kDefaultFontAtlasHeight = 128;
constexpr int kDefaultFontCharsDivisor = 1;
constexpr int kDefaultFontCharsHeight = 10;

struct DefaultFontState {
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;
    std::array<SDL_FRect, kDefaultFontCharsCount> glyph_rects{};
    std::array<int, kDefaultFontCharsCount> glyph_widths{};
    bool glyph_rects_initialized = false;
};

DefaultFontState g_default_font;

constexpr unsigned int kDefaultFontData[512] = {
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00200020, 0x0001b000, 0x00000000, 0x00000000,
    0x8ef92520, 0x00020a00, 0x7dbe8000, 0x1f7df45f, 0x4a2bf2a0, 0x0852091e, 0x41224000, 0x10041450,
    0x2e292020, 0x08220812, 0x41222000, 0x10041450, 0x10f92020, 0x3efa084c, 0x7d22103c, 0x107df7de,
    0xe8a12020, 0x08220832, 0x05220800, 0x10450410, 0xa4a3f000, 0x08520832, 0x05220400, 0x10450410,
    0xe2f92020, 0x0002085e, 0x7d3e0281, 0x107df41f, 0x00200000, 0x8001b000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xc0000fbe, 0xfbf7e00f, 0x5fbf7e7d, 0x0050bee8,
    0x440808a2, 0x0a142fe8, 0x50810285, 0x0050a048, 0x49e428a2, 0x0a142828, 0x40810284, 0x0048a048,
    0x10020fbe, 0x09f7ebaf, 0xd89f3e84, 0x0047a04f, 0x09e48822, 0x0a142aa1, 0x50810284, 0x0048a048,
    0x04082822, 0x0a142fa0, 0x50810285, 0x0050a248, 0x00008fbe, 0xfbf42021, 0x5f817e7d, 0x07d09ce8,
    0x00008000, 0x00000fe0, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x000c0180,
    0xdfbf4282, 0x0bfbf7ef, 0x42850505, 0x004804bf, 0x50a142c6, 0x08401428, 0x42852505, 0x00a808a0,
    0x50a146aa, 0x08401428, 0x42852505, 0x00081090, 0x5fa14a92, 0x0843f7e8, 0x7e792505, 0x00082088,
    0x40a15282, 0x08420128, 0x40852489, 0x00084084, 0x40a16282, 0x0842022a, 0x40852451, 0x00088082,
    0xc0bf4282, 0xf843f42f, 0x7e85fc21, 0x3e0900bf, 0x00000000, 0x00000004, 0x00000000, 0x000c0180,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x04000402, 0x41482000, 0x00000000, 0x00000800,
    0x04000404, 0x4100203c, 0x00000000, 0x00000800, 0xf7df7df0, 0x514bef85, 0xbefbefbe, 0x04513bef,
    0x14414500, 0x494a2885, 0xa28a28aa, 0x04510820, 0xf44145f0, 0x474a289d, 0xa28a28aa, 0x04510be0,
    0x14414510, 0x494a2884, 0xa28a28aa, 0x02910a00, 0xf7df7df0, 0xd14a2f85, 0xbefbe8aa, 0x011f7be0,
    0x00000000, 0x00400804, 0x20080000, 0x00000000, 0x00000000, 0x00600f84, 0x20080000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xac000000, 0x00000f01, 0x00000000, 0x00000000,
    0x24000000, 0x00000f01, 0x00000000, 0x06000000, 0x24000000, 0x00000f01, 0x00000000, 0x09108000,
    0x24fa28a2, 0x00000f01, 0x00000000, 0x013e0000, 0x2242252a, 0x00000f52, 0x00000000, 0x038a8000,
    0x2422222a, 0x00000f29, 0x00000000, 0x010a8000, 0x2412252a, 0x00000f01, 0x00000000, 0x010a8000,
    0x24fbe8be, 0x00000f01, 0x00000000, 0x0ebe8000, 0xac020000, 0x00000f01, 0x00000000, 0x00048000,
    0x0003e000, 0x00000f00, 0x00000000, 0x00008000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000038, 0x8443b80e, 0x00203a03, 0x02bea080, 0xf0000020, 0xc452208a, 0x04202b02,
    0xf8029122, 0x07f0003b, 0xe44b388e, 0x02203a02, 0x081e8a1c, 0x0411e92a, 0xf4420be0, 0x01248202,
    0xe8140414, 0x05d104ba, 0xe7c3b880, 0x00893a0a, 0x283c0e1c, 0x04500902, 0xc4400080, 0x00448002,
    0xe8208422, 0x04500002, 0x80400000, 0x05200002, 0x083e8e00, 0x04100002, 0x804003e0, 0x07000042,
    0xf8008400, 0x07f00003, 0x80400000, 0x04000022, 0x00000000, 0x00000000, 0x80400000, 0x04000002,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00800702, 0x1848a0c2, 0x84010000, 0x02920921,
    0x01042642, 0x00005121, 0x42023f7f, 0x00291002, 0xefc01422, 0x7efdfbf7, 0xefdfa109, 0x03bbbbf7,
    0x28440f12, 0x42850a14, 0x20408109, 0x01111010, 0x28440408, 0x42850a14, 0x2040817f, 0x01111010,
    0xefc78204, 0x7efdfbf7, 0xe7cf8109, 0x011111f3, 0x2850a932, 0x42850a14, 0x2040a109, 0x01111010,
    0x2850b840, 0x42850a14, 0xefdfbf79, 0x03bbbbf7, 0x001fa020, 0x00000000, 0x00001000, 0x00000000,
    0x00002070, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x08022800, 0x00012283, 0x02430802, 0x01010001, 0x8404147c, 0x20000144, 0x80048404, 0x00823f08,
    0xdfbf4284, 0x7e03f7ef, 0x142850a1, 0x0000210a, 0x50a14684, 0x528a1428, 0x142850a1, 0x03efa17a,
    0x50a14a9e, 0x52521428, 0x142850a1, 0x02081f4a, 0x50a15284, 0x4a221428, 0xf42850a1, 0x03efa14b,
    0x50a16284, 0x4a521428, 0x042850a1, 0x0228a17a, 0xdfbf427c, 0x7e8bf7ef, 0xf7efdfbf, 0x03efbd0b,
    0x00000000, 0x04000000, 0x00000000, 0x00000008, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00200508, 0x00840400, 0x11458122, 0x00014210,
    0x00514294, 0x51420800, 0x20a22a94, 0x0050a508, 0x00200000, 0x00000000, 0x00050000, 0x08000000,
    0xfefbefbe, 0xfbefbefb, 0xfbeb9114, 0x00fbefbe, 0x20820820, 0x8a28a20a, 0x8a289114, 0x3e8a28a2,
    0xfefbefbe, 0xfbefbe0b, 0x8a289114, 0x008a28a2, 0x228a28a2, 0x08208208, 0x8a289114, 0x088a28a2,
    0xfefbefbe, 0xfbefbefb, 0xfa2f9114, 0x00fbefbe, 0x00000000, 0x00000040, 0x00000000, 0x00000000,
    0x00000000, 0x00000020, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00210100, 0x00000004, 0x00000000, 0x00000000, 0x14508200, 0x00001402, 0x00000000, 0x00000000,
    0x00000010, 0x00000020, 0x00000000, 0x00000000, 0xa28a28be, 0x00002228, 0x00000000, 0x00000000,
    0xa28a28aa, 0x000022e8, 0x00000000, 0x00000000, 0xa28a28aa, 0x000022a8, 0x00000000, 0x00000000,
    0xa28a28aa, 0x000022e8, 0x00000000, 0x00000000, 0xbefbefbe, 0x00003e2f, 0x00000000, 0x00000000,
    0x00000004, 0x00002028, 0x00000000, 0x00000000, 0x80000000, 0x00003e0f, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
};

constexpr int kCharsWidth[kDefaultFontCharsCount] = {
    3, 1, 4, 6, 5, 7, 6, 2, 3, 3, 5, 5, 2, 4, 1, 7, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 3, 4, 3, 6,
    7, 6, 6, 6, 6, 6, 6, 6, 6, 3, 5, 6, 5, 7, 6, 6, 6, 6, 6, 6, 7, 6, 7, 7, 6, 6, 6, 2, 7, 2, 3, 5,
    2, 5, 5, 5, 5, 5, 4, 5, 5, 1, 2, 5, 2, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 3, 1, 3, 4, 4,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 5, 5, 5, 7, 1, 5, 3, 7, 3, 5, 4, 1, 7, 4, 3, 5, 3, 3, 2, 5, 6, 1, 2, 2, 3, 5, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 3, 3, 3, 3, 7, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 4, 6,
    5, 5, 5, 5, 5, 5, 9, 5, 5, 5, 5, 5, 2, 2, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5,
};

[[noreturn]] void ThrowTextError(const char* message) {
    throw std::runtime_error(std::string(message) + ": " + SDL_GetError());
}

constexpr bool BitCheck(unsigned int value, int bit) {
    return (value & (1u << bit)) != 0u;
}

void EnsureGlyphRects() {
    if (g_default_font.glyph_rects_initialized) {
        return;
    }

    int current_line = 0;
    int current_pos_x = kDefaultFontCharsDivisor;
    int test_pos_x = kDefaultFontCharsDivisor;

    for (int i = 0; i < kDefaultFontCharsCount; ++i) {
        g_default_font.glyph_widths[static_cast<std::size_t>(i)] = kCharsWidth[i];
        SDL_FRect& rect = g_default_font.glyph_rects[static_cast<std::size_t>(i)];
        rect.x = static_cast<float>(current_pos_x);
        rect.y =
            static_cast<float>(kDefaultFontCharsDivisor +
                               current_line * (kDefaultFontCharsHeight + kDefaultFontCharsDivisor));
        rect.w = static_cast<float>(kCharsWidth[i]);
        rect.h = static_cast<float>(kDefaultFontCharsHeight);

        test_pos_x += kCharsWidth[i] + kDefaultFontCharsDivisor;
        if (test_pos_x >= kDefaultFontAtlasWidth) {
            current_line++;
            current_pos_x = 2 * kDefaultFontCharsDivisor + kCharsWidth[i];
            test_pos_x = current_pos_x;
            rect.x = static_cast<float>(kDefaultFontCharsDivisor);
            rect.y = static_cast<float>(kDefaultFontCharsDivisor +
                                        current_line *
                                            (kDefaultFontCharsHeight + kDefaultFontCharsDivisor));
        } else {
            current_pos_x = test_pos_x;
        }
    }

    g_default_font.glyph_rects_initialized = true;
}

int GetGlyphIndex(unsigned char ch) {
    if (ch < 32u) {
        return static_cast<int>('?' - 32);
    }
    return static_cast<int>(ch - 32u);
}

void EnsureFontTexture(SDL_Renderer* renderer) {
    EnsureGlyphRects();
    if (g_default_font.texture != nullptr && g_default_font.renderer == renderer) {
        return;
    }

    if (g_default_font.texture != nullptr) {
        SDL_DestroyTexture(g_default_font.texture);
        g_default_font.texture = nullptr;
    }

    SDL_Surface* surface =
        SDL_CreateSurface(kDefaultFontAtlasWidth, kDefaultFontAtlasHeight, SDL_PIXELFORMAT_RGBA32);
    if (surface == nullptr) {
        ThrowTextError("SDL_CreateSurface failed");
    }

    auto* pixels = static_cast<std::uint32_t*>(surface->pixels);
    const int pixel_count = kDefaultFontAtlasWidth * kDefaultFontAtlasHeight;
    for (int i = 0, counter = 0; i < pixel_count; i += 32) {
        for (int j = 31; j >= 0; --j) {
            pixels[i + j] = BitCheck(kDefaultFontData[counter], j) ? 0xFFFFFFFFu : 0x00FFFFFFu;
        }
        counter++;
    }

    g_default_font.texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_DestroySurface(surface);
    if (g_default_font.texture == nullptr) {
        ThrowTextError("SDL_CreateTextureFromSurface failed");
    }

    SDL_SetTextureBlendMode(g_default_font.texture, SDL_BLENDMODE_BLEND);
    SDL_SetTextureScaleMode(g_default_font.texture, SDL_SCALEMODE_NEAREST);
    g_default_font.renderer = renderer;
}

struct MeasuredText {
    float width = 0.0F;
    float height = 0.0F;
};

MeasuredText measure_textInternal(int font_size, const char* text) {
    const int resolved_font_size = std::max(font_size, kDefaultFontBaseSize);
    const float scale_factor =
        static_cast<float>(resolved_font_size) / static_cast<float>(kDefaultFontBaseSize);
    const float spacing = static_cast<float>(resolved_font_size / kDefaultFontBaseSize);
    const float line_advance =
        static_cast<float>(kDefaultFontBaseSize + (kDefaultFontBaseSize / 2)) * scale_factor;

    float current_width = 0.0F;
    float max_width = 0.0F;
    float total_height = static_cast<float>(resolved_font_size);

    for (const unsigned char* ptr = reinterpret_cast<const unsigned char*>(text); *ptr != '\0';
         ++ptr) {
        const unsigned char ch = *ptr;
        if (ch == '\n') {
            max_width = std::max(max_width, current_width);
            current_width = 0.0F;
            total_height += line_advance;
            continue;
        }

        const int index = GetGlyphIndex(ch);
        current_width +=
            static_cast<float>(g_default_font.glyph_widths[static_cast<std::size_t>(index)]) *
                scale_factor +
            spacing;
    }

    max_width = std::max(max_width, current_width);
    return MeasuredText{max_width, total_height};
}

void draw_textInternal(SDL_Renderer* renderer, int font_size, const char* text, float x, float y,
                       SDL_Color color) {
    EnsureFontTexture(renderer);

    const int resolved_font_size = std::max(font_size, kDefaultFontBaseSize);
    const float scale_factor =
        static_cast<float>(resolved_font_size) / static_cast<float>(kDefaultFontBaseSize);
    const float spacing = static_cast<float>(resolved_font_size / kDefaultFontBaseSize);
    const float line_advance =
        static_cast<float>(kDefaultFontBaseSize + (kDefaultFontBaseSize / 2)) * scale_factor;

    SDL_SetTextureColorMod(g_default_font.texture, color.r, color.g, color.b);
    SDL_SetTextureAlphaMod(g_default_font.texture, color.a);

    float offset_x = 0.0F;
    float offset_y = 0.0F;
    for (const unsigned char* ptr = reinterpret_cast<const unsigned char*>(text); *ptr != '\0';
         ++ptr) {
        const unsigned char ch = *ptr;
        if (ch == '\n') {
            offset_y += line_advance;
            offset_x = 0.0F;
            continue;
        }

        const int index = GetGlyphIndex(ch);
        if (ch != ' ' && ch != '\t') {
            const SDL_FRect& src = g_default_font.glyph_rects[static_cast<std::size_t>(index)];
            const SDL_FRect dst{
                x + offset_x,
                y + offset_y,
                src.w * scale_factor,
                src.h * scale_factor,
            };
            SDL_RenderTexture(renderer, g_default_font.texture, &src, &dst);
        }

        offset_x +=
            static_cast<float>(g_default_font.glyph_widths[static_cast<std::size_t>(index)]) *
                scale_factor +
            spacing;
    }
}

} // namespace

bool init_text_subsystem() {
    EnsureGlyphRects();
    return true;
}

void shutdown_text_subsystem() {
    if (g_default_font.texture != nullptr) {
        SDL_DestroyTexture(g_default_font.texture);
        g_default_font.texture = nullptr;
    }
    g_default_font.renderer = nullptr;
}

void draw_text(SDL_Renderer* renderer, int point_size, const char* text, float x, float y,
               SDL_Color color) {
    draw_textInternal(renderer, point_size, text, x, y, color);
}

bool measure_text(int point_size, const char* text, int* width, int* height) {
    EnsureGlyphRects();
    const MeasuredText measured = measure_textInternal(point_size, text);
    *width = static_cast<int>(measured.width);
    *height = static_cast<int>(measured.height);
    return true;
}

} // namespace vvm
