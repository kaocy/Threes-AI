#pragma once
#include <array>
#include <iostream>
#include <iomanip>
#include "utilities.h"

/**
 * array-based board for threes
 *
 * index (1-d form):
 *  (0)  (1)  (2)
 *  (3)  (4)  (5)
 *
 */
class board {
public:
    typedef uint32_t cell;
    typedef std::array<cell, 3> row;
    typedef std::array<row, 2> grid;
    typedef uint64_t data;
    typedef int reward;
    static const int MAX_TILE = 9;
    static const int MAX_INDEX = 9 * 9 * 9 * 9 * 9 * 9;

public:
    board() : tile(), attr(0) {}
    board(const grid& b, data v = 0) : tile(b), attr(v) {}
    board(const board& b) = default;
    board& operator =(const board& b) = default;

    operator grid&() { return tile; }
    operator const grid&() const { return tile; }
    row& operator [](unsigned i) { return tile[i]; }
    const row& operator [](unsigned i) const { return tile[i]; }
    cell& operator ()(unsigned i) { return tile[i / 3][i % 3]; }
    const cell& operator ()(unsigned i) const { return tile[i / 3][i % 3]; }

    data info() const { return attr; }
    data info(data dat) { data old = attr; attr = dat; return old; }

    int index() const {
        int res = 0;
        for (int i = 0; i < 6; i++) {
            res *= board::MAX_TILE;
            res += tile[i / 3][i % 3];
        }
        return res;
    }

public:
    bool operator ==(const board& b) const { return tile == b.tile; }
    bool operator < (const board& b) const { return tile <  b.tile; }
    bool operator !=(const board& b) const { return !(*this == b); }
    bool operator > (const board& b) const { return b < *this; }
    bool operator <=(const board& b) const { return !(b < *this); }
    bool operator >=(const board& b) const { return !(*this < b); }

public:

    /**
     * place a tile (index value) to the specific position (1-d form index)
     * return 3(tile is 3) or 0(tile is 1, 2) if the action is valid, or -1 if not
     */
    reward place(unsigned pos, cell tile) {
        if (pos >= 6) return -1;
        if (operator()(pos) != 0)   return -1;
        if (tile != 1 && tile != 2 && tile != 3) return -1;
        operator()(pos) = tile;
        return tile == 3 ? 3 : 0;
    }

    /**
     * apply slide to the board
     * return the reward of the action, or -1 if the action is illegal
     */
    reward slide(unsigned opcode) {
        switch (opcode & 0b11) {
        case 0: return slide_up();
        case 1: return slide_right();
        case 2: return slide_down();
        case 3: return slide_left();
        default: return -1;
        }
    }

    reward slide_left() {
        board prev = *this;
        reward score = 0;
        for (int r = 0; r < 2; r++) {
            auto& row = tile[r];
            for (int c = 1; c < 3; c++) {
                int tile = row[c], hold = row[c-1];
                if (tile == 0) continue;
                if (hold) {
                    if (tile > 2 && tile == hold) {
                        score += power(3, tile - 2);
                        row[c-1] = ++tile;
                        row[c] = 0;
                    } else if (tile + hold == 3) {
                        score += 3;
                        row[c-1] = 3;
                        row[c] = 0;
                    }
                } else {
                    row[c-1] = tile;
                    row[c] = 0;
                }
            }
        }
        return (*this != prev) ? score : -1;
    }

    reward slide_right() {
        reflect_horizontal();
        reward score = slide_left();
        reflect_horizontal();
        return score;
    }

    reward slide_up() {
        board prev = *this;
        reward score = 0;
        for (int c = 0; c < 3; c++) {
            cell tile1 = tile[0][c], tile2 = tile[1][c];
            if (tile2 == 0) continue;
            if (tile1 != 0) {
                if (tile1 > 2 && tile1 == tile2) {
                    score += power(3, tile1 - 2);
                    tile[0][c]++;
                    tile[1][c] = 0;
                } else if (tile1 + tile2 == 3) {
                    score += 3;
                    tile[0][c] = 3;
                    tile[1][c] = 0;
                }
            }
            else {
                tile[0][c] = tile[1][c];
                tile[1][c] = 0;
            }
        }
        return (*this != prev) ? score : -1;
    }
    
    reward slide_down() {
        reflect_vertical();
        reward score = slide_up();
        reflect_vertical();
        return score;
    }

    void reflect_horizontal() {
        for (int r = 0; r < 2; r++) {
            std::swap(tile[r][0], tile[r][2]);
        }
    }

    void reflect_vertical() {
        for (int c = 0; c < 3; c++) {
            std::swap(tile[0][c], tile[1][c]);
        }
    }

    void reverse() { reflect_horizontal(); reflect_vertical(); }

public:
    friend std::ostream& operator <<(std::ostream& out, const board& b) {
        for (int i = 0; i < 6; i++) {
            out << std::setw(std::min(i, 1)) << "" << tile_table[b(i)];
        }
        return out;
    }
    friend std::istream& operator >>(std::istream& in, board& b) {
        for (int i = 0; i < 6; i++) {
            while (!std::isdigit(in.peek()) && in.good()) in.ignore(1);
            in >> b(i);
            for (int j = 0; j < 15; j++) {
                if (tile_table[j] == b(i))	b(i) = j;
            }
        }
        return in;
    }

private:
    grid tile;
    data attr;
};
