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
		agent(args), alpha(0.1f), tuple_num(8), tuple_length(4) {
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
		if (meta.find("init") != meta.end()) // pass init=... to initialize the weight
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end()) // pass load=... to load from a specific file
			load_weights(meta["load"]);
		else if (meta.find("init") == meta.end())
			init_weights("0");
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end()) // pass save=... to save to a specific file
			save_weights(meta["save"]);
	}

	virtual void open_episode(const std::string& flag = "") {
		record.clear();
	}

	virtual void close_episode(const std::string& flag = "") {
		after_state last = record[ record.size()-1 ];
		train_weights(last.b, last.b, 0);
		for(int i = record.size() - 2; i >= 0; i--){
			after_state current = record[i];
	        after_state next = record[i + 1];
	        train_weights(current.b, next.b, next.reward);
	    }
	}

	virtual action take_action(const board& b, action prev, int& next_tile) { return action(); }

protected:
	virtual void init_weights(const std::string& info) {
		for (int i=0; i<8; i++)	net.emplace_back(65536);
		// net.emplace_back(65536); // create an empty weight table with size 65536
		// net.emplace_back(65536); // create an empty weight table with size 65536
		// now net.size() == 2; net[0].size() == 65536; net[1].size() == 65536

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
		if (current == next && reward == 0) {
			td_target = 0.0;
		}
		else {
			td_target = reward + state_approximation(next);
		}
		update_value = alpha * (td_target - state_approximation(current));
		for (int i = 0; i < tuple_num; i++) {
			//std::cout << tuple_index(current, i) << " " << net[table_index(i)][tuple_index(current, i)] << std::endl;
			net[i%4][tuple_index(current, i)] += update_value;
			net[i%4+4][tuple_index(current, i, true)] += update_value;
			//net[table_index(i)][tuple_index(current, i, true)] += update_value;
		}
	}

	float state_approximation(const board& b) {
		float value1 = 0.0, value2 = 0.0;
		float horizontal;
		float vertical;
		for (int i = 0; i < 4; i++) {
			value1 += net[i][tuple_index(b, i)];
			value2 += net[i+4][tuple_index(b, i, true)];
		}
		horizontal = std::max(value1, value2);

		value1 = value2 = 0.0;
		for (int i = 4; i < 8; i++) {
			value1 += net[i-4][tuple_index(b, i)];
			value2 += net[i][tuple_index(b, i, true)];
		}
		vertical = std::max(value1, value2);
		return horizontal + vertical;
	}

	int table_index(int index) {
		if (index == 0 || index == 3)	return 0;
		if (index == 4 || index == 7)	return 0;
		else	return 1;
	}

	int tuple_index(board b, int index, bool flag=false) {
		int result = 0;
		if (index >= 4)	b.rotate_left();
		index %= 4;
		if (flag) for (int i = tuple_length-1; i >=0; i--) {
			int tile = b[index][i];		
			result <<= 4;
			result |= tile;
		}
		else for (int i = 0; i < tuple_length; i++) {
			int tile = b[index][i];		
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
		for (int op : opcode) {
			board tmp = board(before);
			board::reward reward = tmp.slide(op);		
			if (reward != -1) {
				float value = reward + state_approximation(tmp);
				if (value > best_value) {
					best_value = value;
					best_reward = reward;
					best_op = op;
					best_state = tmp;
				}
			}
		}
		//std::cout << best_reward << " " << best_value << std::endl;
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
		bag({ 1, 2, 3 }),
		tile_bag(0b1110),
		popup(1, 3) {}

	virtual void open_episode(const std::string& flag = "") { tile_bag = 0b1110; } 

	virtual action take_action(const board& after, action prev, int& next_tile) {
		std::shuffle(space.begin(), space.end(), engine);
		std::shuffle(bag.begin(), bag.end(), engine);

		board::cell tile = next_tile;
		// for the first place
		if (tile == 0) {
			tile = popup(engine);
			tile_bag ^= (1 << tile);
		}
		// choose next tile
		for (int t : bag) {
			if (tile_bag & (1 << t)) {
				next_tile = t;
				tile_bag ^= (1 << next_tile);
				if (tile_bag == 0)	tile_bag = 0b1110;
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
				if(slide_op == 0 && pos < 12)		continue;
				if(slide_op == 1 && pos % 4 != 0)	continue;
				if(slide_op == 2 && pos > 3)		continue;
				if(slide_op == 3 && pos % 4 != 3)	continue;
				if (after(pos) != 0) continue;				
				return action::place(pos, tile);
			}			
		}		
		return action();
	}

private:
	std::array<int, 16> space;
	std::array<int, 3> bag;
	int tile_bag;
	std::uniform_int_distribution<int> popup;
};
