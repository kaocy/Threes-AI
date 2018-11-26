#pragma once

// for tile conversion
uint32_t tile_table[15] = { 0, 1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144 };

int power(int x, int y) {
    if(y <= 0)	return 1;
    if(y & 1)	return x * power(x, y - 1);
    int temp = power(x, y >> 1);
    return temp * temp;
}