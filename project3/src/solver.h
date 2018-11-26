#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>
#include <cfloat>
#include "board.h"
#include "action.h"
#include "utilities.h"

class state_type {
public:
    enum type : char {
        before  = 'b',
        after   = 'a',
        illegal = 'i'
    };

public:
    state_type() : t(illegal) {}
    state_type(const state_type& st) = default;
    state_type(state_type::type code) : t(code) {}

    friend std::istream& operator >>(std::istream& in, state_type& type) {
        std::string s;
        if (in >> s) type.t = static_cast<state_type::type>((s + " ").front());
        return in;
    }

    friend std::ostream& operator <<(std::ostream& out, const state_type& type) {
        return out << char(type.t);
    }

    bool is_before()  const { return t == before; }
    bool is_after()   const { return t == after; }
    bool is_illegal() const { return t == illegal; }

private:
    type t;
};

class state_hint {
public:
    state_hint(const board& state) : state(const_cast<board&>(state)) {}

    char type() const { return state.info() ? state.info() + '0' : 'x'; }
    operator board::cell() const { return state.info(); }

public:
    friend std::istream& operator >>(std::istream& in, state_hint& hint) {
        while (in.peek() != '+' && in.good()) in.ignore(1);
        char v; in.ignore(1) >> v;
        hint.state.info(v != 'x' ? v - '0' : 0);
        return in;
    }
    friend std::ostream& operator <<(std::ostream& out, const state_hint& hint) {
        return out << "+" << hint.type();
    }

private:
    board& state;
};


class solver {
public:
    typedef float value_t;

public:
    class answer {
    public:
        answer(value_t min = 0.0/0.0, value_t avg = 0.0/0.0, value_t max = 0.0/0.0) : min(min), avg(avg), max(max) {}
        answer operator +(const answer& a) {
            return answer(min + a.min, avg + a.avg, max + a.max);
        }
        answer operator +(float f) {
            return answer(min + f, avg + f, max + f);
        }
        friend std::ostream& operator <<(std::ostream& out, const answer& ans) {
            return !std::isnan(ans.avg) ? (out << ans.min << " " << ans.avg << " " << ans.max) : (out << "-1") ;
        }
    public:
        value_t min, avg, max;
    };

public:
    solver(const std::string& args) {
        // TODO: explore the tree and save the result

        table_b.resize(board::MAX_INDEX);
        for (int i = 0; i < board::MAX_INDEX; i++) {
            table_b[i].resize(3);
        }

        table_a.resize(board::MAX_INDEX);
        for (int i = 0; i < board::MAX_INDEX; i++) {
            table_a[i].resize(3);
            for (int j = 0; j < 3; j++) {
                table_a[i][j].resize(4);
            }
        }

        for (int pos = 0; pos < 6; pos++) {
            for (int tile = 1; tile <= 3; tile++) {
                for (int hint = 1; hint <= 3; hint++) {
                    if (tile == hint)   continue;
                    board b;
                    b.info(hint);
                    action::place(pos, tile).apply(b);
                    before_value(b, 0b1110 ^ (1 << hint) ^ (1 << tile));
                }
            }
        }
    }

    answer solve(const board& state, state_type type = state_type::before) {
        // TODO: find the answer in the lookup table and return it
        //       do NOT recalculate the tree at here

        int index = state.index();
        board::cell hint = state_hint(state) - 1;

        for (int i = 0; i < 6; i++) {
            if (state(i) >= board::MAX_TILE)
                return {};
        }

        if (index >= board::MAX_INDEX)
            return {};

        if (type.is_before() && !std::isnan(table_b[index][hint].avg))
            return table_b[index][hint];

        if (type.is_after()) {
            for (int op : { 0, 1, 2, 3 }) {
                if (!std::isnan(table_a[index][hint][op].avg))
                    return table_a[index][hint][op];
            }
        }

        return {};
    }

private:
    answer before_value(const board& b, int tile_bag) {
        int index = b.index();
        board::cell hint = state_hint(b) - 1;

        if (!std::isnan(table_b[index][hint].avg))
            return table_b[index][hint];

        value_t best_avg = FLT_MIN;
        value_t min = 0.0, avg = 0.0, max = 0.0;

        for (int op : { 0, 1, 2, 3 }) {
            board temp(b);
            board::reward reward = temp.slide(op);
            if (reward != -1) {
                // std::cout << op << std::endl;
                answer value = after_value(temp, tile_bag, op);
                if (value.avg + reward > best_avg) {
                    best_avg = value.avg + reward;
                    min = value.min + reward;
                    avg = value.avg + reward;
                    max = value.max + reward;
                }
            }
        }

        if (best_avg == FLT_MIN)
            return table_b[index][hint] = answer(0.0, 0.0, 0.0);
        else
            return table_b[index][hint] = answer(min, avg, max);
    }

    answer after_value(const board& b, int tile_bag, int last_op) {
        int index = b.index();
        board::cell hint = state_hint(b) - 1;

        if (!std::isnan(table_a[index][hint][last_op].avg))
            return table_a[index][hint][last_op];

        if (tile_bag == 0)  tile_bag = 0b1110;

        int count = 0;
        value_t avg = 0.0;
        value_t min = FLT_MAX;
        value_t max = FLT_MIN;

        for (int pos = 0; pos < 6; pos++) {
            if ((last_op == 0) && (pos < 3))        continue;
            if ((last_op == 1) && (pos % 3 != 0))   continue;
            if ((last_op == 2) && (pos > 2))        continue;
            if ((last_op == 3) && (pos % 3 != 2))   continue;

            for (int tile = 1; tile <= 3; tile++) {
                if (tile_bag & (1 << tile)) {
                    board temp(b);
                    temp.info(tile);
                    board::reward reward = action::place(pos, hint + 1).apply(temp);

                    if (reward != -1) {
                        count++;
                        // std::cout << pos << " " << hint + 1 << std::endl;
                        answer value = before_value(temp, tile_bag ^ (1 << tile));
                        min = std::min(min, value.min + reward);
                        max = std::max(max, value.max + reward);
                        avg += value.avg + reward;
                    }
                }
            }
        }

        return table_a[index][hint][last_op] = answer(min, avg / count, max);
    }

private:
    // TODO: place your transposition table here
    std::vector< std::vector<answer> > table_b;
    std::vector< std::vector< std::vector<answer> > > table_a;
};
