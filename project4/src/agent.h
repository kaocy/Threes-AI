#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <fstream>

class agent {
public:
    agent(const std::string& args = "") {
        std::stringstream ss("name=unknown role=unknown " + args);
        for (std::string pair; ss >> pair; ) {
            std::string key = pair.substr(0, pair.find('='));
            std::string value = pair.substr(pair.find('=') + 1);
            meta[key] = { value };
        }
    }
    virtual ~agent() {}
    virtual void open_episode(const std::string& flag = "") {}
    virtual void close_episode(const std::string& flag = "") {}
    virtual action take_action(const board& b, action prev, int& next_tile) { return action(); }
    virtual bool check_for_win(const board& b) { return false; }

public:
    virtual std::string property(const std::string& key) const { return meta.at(key); }
    virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
    virtual std::string name() const { return property("name"); }
    virtual std::string role() const { return property("role"); }

protected:
    typedef std::string key;
    struct value {
        std::string value;
        operator std::string() const { return value; }
        template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
        operator numeric() const { return numeric(std::stod(value)); }
    };
    std::map<key, value> meta;
};

/**
 * base agent for agents with weight tables
 */
class weight_agent : public agent {
public:
    weight_agent(const std::string& args = "") : 
        agent(args), alpha(0.003125f), tuple_num(4), tuple_length(6) {
        if (meta.find("alpha") != meta.end())
            alpha = float(meta["alpha"]);
        if (meta.find("init") != meta.end()) // pass init=... to initialize the weight
            init_weights(meta["init"]);
        if (meta.find("load") != meta.end()) // pass load=... to load from a specific file
            load_weights(meta["load"]);
        else if (meta.find("init") == meta.end())
            init_weights("0");

        indices.push_back({0, 4, 8, 12, 9, 13});
        indices.push_back({1, 5, 9, 13, 10, 14});
        indices.push_back({1, 5, 9, 2, 6, 10});
        indices.push_back({2, 6, 10, 3, 7, 11});
    }
    virtual ~weight_agent() {
        if (meta.find("save") != meta.end()) // pass save=... to save to a specific file
            save_weights(meta["save"]);
    }

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

    virtual action take_action(const board& b, action prev, int& next_tile) { return action(); }

    virtual void reduce_learning_rate() { alpha /= 2.0; }

protected:
    virtual void init_weights(const std::string& info) {
        for (int i = 0; i < tuple_num; i++)
            net.emplace_back(1 << 24); // create an empty weight table with size 16^6
    }
    virtual void load_weights(const std::string& path) {
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

    virtual void train_weights(const board& current, const board& next, const int reward = 0) {
        float td_target, update_value;

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
            for (int i = 0; i < tuple_num; i++) net[i][tuple_index(tmp, i)] += update_value;
            tmp.reflect_vertical();
            for (int i = 0; i < tuple_num; i++) net[i][tuple_index(tmp, i)] += update_value;
            tmp.reflect_vertical();
        }
    }

    float state_approximation(const board& b) {
        float value = 0.0;
        board tmp(b);
        for (int k = 0; k < 4; k++) {
            if (k > 0)  tmp.rotate_right();
            for (int i = 0; i < tuple_num; i++) value += net[i][tuple_index(tmp, i)];
            tmp.reflect_vertical();
            for (int i = 0; i < tuple_num; i++) value += net[i][tuple_index(tmp, i)];
            tmp.reflect_vertical();
        }
        return value / 8.0;
    }

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

protected:
    struct after_state {
        board b;
        int reward;
        after_state(board b = {}, int reward = 0) : b(b), reward(reward) {}
    };
    std::vector<after_state> record;

protected:
    std::vector<std::vector<int>> indices;
    std::vector<weight> net;
    float alpha;
    int tuple_num;
    int tuple_length;
};

/**
 * dummy player
 * select a legal action with up and right slide in priority
 */
class player : public weight_agent {
public:
    player(const std::string& args = "") : 
        weight_agent("name=dummy role=player " + args),
        opcode({ 0, 1, 2, 3 }) {}

    virtual action take_action(const board& before, action prev, int& next_tile) {
        float best_value = -999999999.0;
        int best_op = -1, best_reward;
        board best_state;

        // choose the best slide op
        for (int op : opcode) {
            board tmp = board(before);
            board::reward immediate_reward = tmp.slide(op);
            if (immediate_reward != -1) {
                float value = immediate_reward + state_approximation(tmp);
                if (value > best_value) {
                    best_value = value;
                    best_reward = immediate_reward;
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
    std::array<int, 4> opcode;
};


class random_agent : public agent {
public:
    random_agent(const std::string& args = "") : agent(args) {
        if (meta.find("seed") != meta.end())
            engine.seed(int(meta["seed"]));
    }
    virtual ~random_agent() {}

protected:
    std::default_random_engine engine;
};

/**
 * random environment
 * add a new random tile to an empty cell from tile bag
 * tile bag contain 1, 2, 3 tile
 * once the tile bag is empty, reset it
 */
class rndenv : public random_agent {
public:
    rndenv(const std::string& args = "") : 
        random_agent("name=random role=environment " + args),
        space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }),
        bag({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }),
        tile_bag((1 << 12) - 1),
        popup(0, 11) {}

    virtual void open_episode(const std::string& flag = "") { tile_bag = (1 << 12) - 1; } 

    virtual action take_action(const board& after, action prev, int& next_tile) {
        std::shuffle(space.begin(), space.end(), engine);
        std::shuffle(bag.begin(), bag.end(), engine);

        board::cell tile = next_tile;
        // for the first place
        if (tile == 0) {
            tile = popup(engine);
            tile_bag ^= (1 << tile);
            tile = tile / 4 + 1;
        }
        // choose next tile
        for (int t : bag) {
            if (tile_bag & (1 << t)) {
                next_tile = t / 4 + 1;
                tile_bag ^= (1 << t);
                if (tile_bag == 0)  tile_bag = (1 << 12) - 1;
                break;
            }
        }
        // choose position for first 9 place
        if (prev.type() == action::place::type) {
            for (int pos : space) {
                if (after(pos) != 0) continue;
                return action::place(pos, tile);
            }
        }
        // choose position for place after slide
        else {
            int slide_op = prev.event() & 0b11;
            for (int pos : space) {
                if(slide_op == 0 && pos < 12)       continue;
                if(slide_op == 1 && pos % 4 != 0)   continue;
                if(slide_op == 2 && pos > 3)        continue;
                if(slide_op == 3 && pos % 4 != 3)   continue;
                if (after(pos) != 0) continue;
                return action::place(pos, tile);
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
