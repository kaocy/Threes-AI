#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include <cfloat>
#include "board.h"
#include "action.h"
#include "weight.h"

const int tuple_num = 4;
const int tuple_length = 6;
std::vector<std::vector<int>> indices;
std::vector<weight> net;

class agent {
public:
    agent(const std::string& args = "") {
        std::stringstream ss("name=unknown role=unknown " + args);
        for (std::string pair; ss >> pair; ) {
            std::string key = pair.substr(0, pair.find('='));
            std::string value = pair.substr(pair.find('=') + 1);
            meta[key] = { value };
        }
        if (meta.find("seed") != meta.end())
            engine.seed(int(meta["seed"]));

        if (indices.size() == 0) {
            indices.push_back({0, 4, 8, 12, 9, 13});
            indices.push_back({1, 5, 9, 13, 10, 14});
            indices.push_back({1, 5, 9, 2, 6, 10});
            indices.push_back({2, 6, 10, 3, 7, 11});
        }
    }
    virtual ~agent() {}
    virtual void open_episode(const std::string& flag = "") {}
    virtual void close_episode(const std::string& flag = "") {}
    virtual action take_action(board& b, action prev) { return action(); }
    virtual bool check_for_win(const board& b) { return false; }

public:
    virtual std::string property(const std::string& key) const { return meta.at(key); }
    virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
    virtual std::string name() const { return property("name"); }
    virtual std::string role() const { return property("role"); }

protected:
    virtual void init_weights() {
        if (net.size() > 0) return ;
        for (int i = 0; i < tuple_num * 4; i++)
            net.emplace_back(1 << 24); // create an empty weight table with size 16^6 * 4 hint tile
    }
    virtual void load_weights(const std::string& path) {
        if (net.size() > 0) return ;
        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (!in.is_open()) std::exit(-1);
        uint32_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        net.resize(size);
        for (weight& w : net) in >> w;
        in.close();
    }
    virtual void save_weights(const std::string& path) {
        std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!out.is_open()) std::exit(-1);
        uint32_t size = net.size();
        out.write(reinterpret_cast<char*>(&size), sizeof(size));
        for (weight& w : net) out << w;
        out.close();
    }

protected:
    // return the tuple index in weight table
    int tuple_index(const board& b, int index) {
        int result = 0;
        for (int i = 0; i < tuple_length; i++) {
            int x = indices[index][i];
            int tile = b[x / 4][x % 4];
            result <<= 4;
            result |= tile;
        }
        return result;
    }

    float state_approximation(const board& b) {
        float value = 0.0;
        // hint tile index in weight table is 1, 2, 3, 0 for 1-tile, 2-tile, 3-tile, bonus-tile
        int hint = b.info() > 3 ? 0 : b.info();
        board tmp(b);
        for (int k = 0; k < 4; k++) {
            if (k > 0)  tmp.rotate_right();
            for (int i = 0; i < tuple_num; i++) value += net[i * 4 + hint][tuple_index(tmp, i)];
            tmp.reflect_vertical();
            for (int i = 0; i < tuple_num; i++) value += net[i * 4 + hint][tuple_index(tmp, i)];
            tmp.reflect_vertical();
        }
        return value / 8.0;
    }

    // return the worst board value
    float after_value(const board& after, int last_op, int level) {
        if (level == 1)
            return state_approximation(after);

        board::cell hint = after.info();       
        if (hint > 3) {
            // randomly guess next bonus tile
            std::uniform_int_distribution<int> popup_bonus(4, after.get_largest() - 3);
            hint = popup_bonus(engine);
        }

        // randomly pick one hint tile for search
        int next_hint_tile = 0;
        std::uniform_int_distribution<int> popup1(0, 20);
        if (after.can_place_bonus_tile() && popup1(engine) == 0) {
            next_hint_tile = 4;
        }
        else {
            std::uniform_int_distribution<int> popup2(1, 3);
            next_hint_tile = popup2(engine);
        }

        float worst_value = FLT_MAX;
        for (int pos = 0; pos < 16; pos++) {
            if ((last_op == 0) && (pos < 12))       continue;
            if ((last_op == 1) && (pos % 4 != 0))   continue;
            if ((last_op == 2) && (pos > 3))        continue;
            if ((last_op == 3) && (pos % 4 != 3))   continue;
            if (after(pos) != 0) continue;

            board tmp = board(after);
            tmp.info(next_hint_tile);
            board::reward reward = tmp.place(pos, hint);
            if (reward != -1) {
                float value = reward + before_value(tmp, level - 1);
                if (value < worst_value) {
                    worst_value = value;
                }
            }
        }
        return worst_value;
    }

    // return the best board value
    float before_value(const board& before, int level) {
        float best_value = -FLT_MAX;
        for (int op : {0, 1, 2, 3}) {
            board tmp(before);
            board::reward reward = tmp.slide(op);
            if (reward != -1) {
                float value = reward + after_value(tmp, op, level - 1);
                if (value > best_value) {
                    best_value = value;
                }
            }
        }
        return best_value != -FLT_MAX ? best_value : 0.0;
    }

protected:
    typedef std::string key;
    struct value {
        std::string value;
        operator std::string() const { return value; }
        template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
        operator numeric() const { return numeric(std::stod(value)); }
    };
    std::map<key, value> meta;
    std::default_random_engine engine;
};

/**
 * learning player
 * select a legal action with best value
 */
class player : public agent {
public:
    player(const std::string& args = "") :
        agent("name=learning role=player " + args),
        opcode({ 0, 1, 2, 3 }),
        alpha(0.003125f) {
        if (meta.find("alpha") != meta.end())
            alpha = float(meta["alpha"]);
        if (meta.find("load") != meta.end()) // pass load=... to load from a specific file
            load_weights(meta["load"]);
        else
            init_weights();
    }
    ~player() {
        if (meta.find("save") != meta.end()) // pass save=... to save to a specific file
            save_weights(meta["save"]);
    }

public:
    virtual void open_episode(const std::string& flag = "") {
        record.clear();
    }
    virtual void close_episode(const std::string& flag = "") {
        if (record.size() <= 0) return ;

        after_state last = record[ record.size()-1 ];
        train_weights(last.b, last.b, 0);
        for(int i = record.size() - 2; i >= 0; i--){
            after_state current = record[i];
            after_state next = record[i + 1];
            train_weights(current.b, next.b, next.reward);
        }
    }
    virtual void reduce_learning_rate() { alpha *= 0.75; }

private:
    void train_weights(const board& current, const board& next, const int reward = 0) {
        float td_target, update_value;
        int hint = current.info() > 3 ? 0 : current.info();

        // for the final state
        if (current == next && reward == 0) {
            td_target = 0.0;
        }
        else {
            td_target = reward + state_approximation(next);
        }
        update_value = alpha * (td_target - state_approximation(current));

        board tmp(current);
        for (int k = 0; k < 4; k++) {
            if (k > 0)  tmp.rotate_right();
            for (int i = 0; i < tuple_num; i++) net[i * 4 + hint][tuple_index(tmp, i)] += update_value;
            tmp.reflect_vertical();
            for (int i = 0; i < tuple_num; i++) net[i * 4 + hint][tuple_index(tmp, i)] += update_value;
            tmp.reflect_vertical();
        }
    }

public:
    virtual action take_action(board& before, action prev) {
        float best_value = -FLT_MAX;
        int best_op = -1, best_reward;
        board best_state;

        // choose the best slide op
        for (int op : opcode) {
            board tmp = board(before);
            board::reward reward = tmp.slide(op);
            if (reward != -1) {
                float value = reward + after_value(tmp, op, 3);
                if (value > best_value) {
                    best_value = value;
                    best_reward = reward;
                    best_op = op;
                    best_state = tmp;
                }
            }
        }
        if (best_op != -1) {
            record.emplace_back(best_state, best_reward);
            return action::slide(best_op);
        }
        return action();
    }

private:
    struct after_state {
        board b;
        int reward;
        after_state(board b = {}, int reward = 0) : b(b), reward(reward) {}
    };
    std::vector<after_state> record;

private:
    std::array<int, 4> opcode;
    float alpha;
};

/**
 * random environment
 * add a new random tile to an empty cell from tile bag
 * tile bag contain 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 tile
 * once the tile bag is empty, reset it
 * with 1/21 probability to place bonus tile
 * choose worst position to let player get less score
 */
class rndenv : public agent {
public:
    rndenv(const std::string& args = "") :
        agent("name=random role=environment " + args),
        space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }),
        bag({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }),
        tile_bag((1 << 12) - 1),
        popup(0, 20) {
        if (meta.find("load") != meta.end()) // pass load=... to load from a specific file
            load_weights(meta["load"]);
        else
            init_weights();
    }

    virtual void open_episode(const std::string& flag = "") { tile_bag = (1 << 12) - 1; }

    virtual action take_action(board& after, action prev) {
        std::shuffle(space.begin(), space.end(), engine);
        std::shuffle(bag.begin(), bag.end(), engine);

        board::cell tile = after.info();
        // for the first place
        if (tile == 0) {
            tile = bag[0];
            tile_bag ^= (1 << tile);
            tile = tile / 4 + 1;
            after.add_tile();
        }

        // for first 9 place
        if (prev.type() == action::place::type) {
            // choose hint tile
            for (int t : bag) {
                if (tile_bag & (1 << t)) {
                    after.info(t / 4 + 1);
                    tile_bag ^= (1 << t);
                    if (tile_bag == 0)  tile_bag = (1 << 12) - 1;
                    break;
                }
            }
            after.add_tile();

            for (int pos : space) {
                if (after(pos) != 0) continue;
                return action::place(pos, tile);
            }
        }
        // for place after slide
        else {
            if (tile > 3) {
                // randomly choose bonus tile: 6-tile to (Vmax/8)-tile
                std::uniform_int_distribution<int> popup_bonus(4, after.get_largest() - 3);
                tile = popup_bonus(engine);
            }

            int slide_op = prev.event() & 0b11;
            // for training
            if (name() == "random") {
                // choose hint tile, with 1/21 probability to place bonus tile
                if (after.can_place_bonus_tile() && popup(engine) == 0) {
                    after.info(4);
                    after.add_bonus_tile();
                    after.add_tile();
                }
                else {
                    for (int t : bag) {
                        if (tile_bag & (1 << t)) {
                            after.info(t / 4 + 1);
                            tile_bag ^= (1 << t);
                            if (tile_bag == 0)  tile_bag = (1 << 12) - 1;
                            break;
                        }
                    }
                    after.add_tile();
                }

                // randomly choose one legal position
                for (int pos : space) {
                    if(slide_op == 0 && pos < 12)       continue;
                    if(slide_op == 1 && pos % 4 != 0)   continue;
                    if(slide_op == 2 && pos > 3)        continue;
                    if(slide_op == 3 && pos % 4 != 3)   continue;
                    if (after(pos) != 0) continue;
                    return action::place(pos, tile);
                }
            }
            // for playing, choose the worst position and hint tile to minimize score
            else {
                float worst_value = FLT_MAX;
                int worst_pos = -1;

                // choose hint tile, with 1/21 probability to place bonus tile
                if (after.can_place_bonus_tile() && popup(engine) == 0) {
                    after.info(4);
                    after.add_bonus_tile();
                    after.add_tile();

                    for (int pos : space) {
                        if(slide_op == 0 && pos < 12)       continue;
                        if(slide_op == 1 && pos % 4 != 0)   continue;
                        if(slide_op == 2 && pos > 3)        continue;
                        if(slide_op == 3 && pos % 4 != 3)   continue;
                        if (after(pos) != 0) continue;

                        board tmp = board(after);
                        board::reward reward = tmp.place(pos, tile);
                        if (reward != -1) {
                            float value = reward + before_value(tmp, 2);
                            if (value < worst_value) {
                                worst_value = value;
                                worst_pos = pos;
                            }
                        }
                    }
                }
                else {
                    int worst_hint = -1;
                    after.add_tile();
                    // choose the worst hint tile to player
                    for (int t : bag) if (tile_bag & (1 << t)) {
                        for (int pos : space) {
                            if(slide_op == 0 && pos < 12)       continue;
                            if(slide_op == 1 && pos % 4 != 0)   continue;
                            if(slide_op == 2 && pos > 3)        continue;
                            if(slide_op == 3 && pos % 4 != 3)   continue;
                            if (after(pos) != 0) continue;

                            board tmp = board(after);
                            board::reward reward = tmp.place(pos, tile);
                            tmp.info(t / 4 + 1);
                            if (reward != -1) {
                                float value = reward + before_value(tmp, 2);
                                if (value < worst_value) {
                                    worst_value = value;
                                    worst_pos = pos;
                                    worst_hint = t; // 0~11
                                }
                            }
                        }
                    }
                    if (worst_hint != -1) {
                        after.info(worst_hint / 4 + 1);
                        tile_bag ^= (1 << worst_hint);
                        if (tile_bag == 0)  tile_bag = (1 << 12) - 1;
                    }
                }

                if (worst_pos != -1) {
                    return action::place(worst_pos, tile);
                }
            }
        }
        return action();
    }

private:
    std::array<int, 16> space;
    std::array<int, 12> bag;
    int tile_bag;
    std::uniform_int_distribution<int> popup;
};
