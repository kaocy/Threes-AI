#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"

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
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
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
		// for first 9 place
		if (prev.type() == action::place::type) {
			for (int pos : space) {
				if (after(pos) != 0) continue;
				return action::place(pos, tile);
			}
		}
		// for place after slide
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

/**
 * dummy player
 * select a legal action randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=dummy role=player " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before, action prev, int& next_tile) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};
