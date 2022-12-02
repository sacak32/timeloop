/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <iterator>
#include <unordered_set>
#include <boost/functional/hash.hpp>

#include "mapping/mapping.hpp"
#include "mapspaces/mapspace-base.hpp"
#include "util/misc.hpp"
#include "search/search.hpp"

#define MAX_ITER_COUNT 5

namespace search
{

class GeneticSearch : public SearchAlgorithm
{
 private:
  enum class State
  {
    Init,
    Random,
    SelfMutate,
    Terminated
  };

  // Config.
  mapspace::MapSpace* mapspace_;
  std::uint32_t nGenerations_;
  std::uint32_t population_size_;
  std::uint32_t tournament_size_;
  std::double_t p_crossover_;
  std::double_t p_loop_;
  std::double_t p_data_bypass_;
  std::double_t p_index_factorization_;
  std::double_t p_random_;

  // std::unordered_set<std::uint64_t> bad_;
  std::unordered_set<uint128_t> visited_;
  bool filter_revisits_;

  // Submodules.
  std::array<PatternGenerator128*, int(mapspace::Dimension::Num)> pgens_;
  
  // Live state.
  State state_;
  mapspace::ID mapping_id_;
  uint128_t masking_space_covered_;
  uint128_t valid_mappings_;

  // Current Generation
  uint32_t cur_gen = 1;

  // Worklist
  std::vector<std::pair<mapspace::ID,double>> current_worklist_;
  std::vector<std::pair<mapspace::ID,double>> next_worklist_;

  bool flag;

  // Roll the dice along a single mapspace dimension.
  void Roll(mapspace::Dimension dim);
  void selfMutate(mapspace::ID& id1,mapspace::ID& id2);

  // Flags
  bool isInitial;
  
 public:
  GeneticSearch(config::CompoundConfigNode config, 
                mapspace::MapSpace* mapspace,
                std::uint32_t nGenerations_,
                std::uint32_t population_size_,
                std::uint32_t tournament_size_,
                std::double_t p_crossover_,
                std::double_t p_loop_,
                std::double_t p_data_bypass_,
                std::double_t p_index_factorization_,
                std::double_t p_random_
                );

  // This class does not support being copied
  GeneticSearch(const GeneticSearch&) = delete;
  GeneticSearch& operator=(const GeneticSearch&) = delete;

  ~GeneticSearch();
  
  bool Next(mapspace::ID& mapping_id);

  void Report(Status status, double cost = 0);
};

} // namespace search
