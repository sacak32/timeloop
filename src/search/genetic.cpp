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

#include "search/genetic.hpp"

namespace search
{

void GeneticSearch::Roll(mapspace::Dimension dim)
{
  mapping_id_.Set(int(dim), pgens_[int(dim)]->Next());
}

GeneticSearch::GeneticSearch(config::CompoundConfigNode config, mapspace::MapSpace* mapspace) :
    SearchAlgorithm(),
    mapspace_(mapspace),
    state_(State::Random),
    mapping_id_(mapspace->AllSizes()),
    masking_space_covered_(mapspace_->Size(mapspace::Dimension::DatatypeBypass)),
    valid_mappings_(0),
    iter_count(0)
{
  filter_revisits_ = false;
  config.lookupValue("filter-revisits", filter_revisits_);    

  pgens_[int(mapspace::Dimension::IndexFactorization)] =
    new RandomGenerator128(mapspace_->Size(mapspace::Dimension::IndexFactorization));
  pgens_[int(mapspace::Dimension::LoopPermutation)] =
    new RandomGenerator128(mapspace_->Size(mapspace::Dimension::LoopPermutation));
  pgens_[int(mapspace::Dimension::Spatial)] =
    new RandomGenerator128(mapspace_->Size(mapspace::Dimension::Spatial));
  pgens_[int(mapspace::Dimension::DatatypeBypass)] =
    new SequenceGenerator128(mapspace_->Size(mapspace::Dimension::DatatypeBypass));
  // std::cout << "Mapping ID base dimensions = " << mapping_id_.Base().size() << std::endl;
  // std::cout << "Mapping ID base = ";
  // Print<>(mapping_id_.Base());
  // std::cout << std::endl;

  // Special case: if the index factorization space has size 0
  // (can happen with residual mapspaces) then we init in terminated
  // state.
  if (mapspace_->Size(mapspace::Dimension::IndexFactorization) == 0)
    state_ = State::Terminated;
}

GeneticSearch::~GeneticSearch()
{
  delete static_cast<RandomGenerator128*>(
    pgens_[int(mapspace::Dimension::IndexFactorization)]);
  delete static_cast<RandomGenerator128*>(
    pgens_[int(mapspace::Dimension::LoopPermutation)]);
  delete static_cast<RandomGenerator128*>(
    pgens_[int(mapspace::Dimension::Spatial)]);
  delete static_cast<SequenceGenerator128*>(
    pgens_[int(mapspace::Dimension::DatatypeBypass)]);
}

// Self mutate, randomly selects a new loop permutation
void GeneticSearch::selfMutate(mapspace::ID& id1,mapspace::ID& id2) {
  // (void) id;

  // mapspace::ID id1 = bestlist_.front.first;
  // mapspace::ID id2 = id;

  uint128_t mapping_index_factorization_id = id1[int(mapspace::Dimension::IndexFactorization)];  
  uint128_t mapping_permutation_id = id1[int(mapspace::Dimension::LoopPermutation)];
  uint128_t mapping_spatial_id = id1[int(mapspace::Dimension::Spatial)];
  uint128_t mapping_datatype_bypass_id = id1[int(mapspace::Dimension::DatatypeBypass)];

  std::cout << "ID1: ";
  std::cout << mapping_index_factorization_id << " ";
  std::cout << mapping_permutation_id << " ";
  std::cout << mapping_spatial_id << " ";
  std::cout << mapping_datatype_bypass_id << std::endl;

  mapping_index_factorization_id = id2[int(mapspace::Dimension::IndexFactorization)];  
  mapping_permutation_id = id2[int(mapspace::Dimension::LoopPermutation)];
  mapping_spatial_id = id2[int(mapspace::Dimension::Spatial)];
  mapping_datatype_bypass_id = id2[int(mapspace::Dimension::DatatypeBypass)];

  std::cout << "ID2: ";
  std::cout << mapping_index_factorization_id << " ";
  std::cout << mapping_permutation_id << " ";
  std::cout << mapping_spatial_id << " ";
  std::cout << mapping_datatype_bypass_id << std::endl;

  mapping_id_.Set(int(mapspace::Dimension::IndexFactorization), id1[int(mapspace::Dimension::IndexFactorization)]);
  mapping_id_.Set(int(mapspace::Dimension::LoopPermutation), id2[int(mapspace::Dimension::LoopPermutation)]);
  mapping_id_.Set(int(mapspace::Dimension::Spatial), id2[int(mapspace::Dimension::Spatial)]);
  mapping_id_.Set(int(mapspace::Dimension::DatatypeBypass), id1[int(mapspace::Dimension::DatatypeBypass)]);
}

bool GeneticSearch::Next(mapspace::ID& mapping_id)
{
  if (state_ == State::Terminated)
  {
    return false;
  }

  // Initial worklist entirely consist of random mappings
  if (state_ == State::Random) {
    // Roll new mapping
    Roll(mapspace::Dimension::IndexFactorization);
    Roll(mapspace::Dimension::LoopPermutation);
    Roll(mapspace::Dimension::Spatial);
    Roll(mapspace::Dimension::DatatypeBypass);
  } else if (state_ == State::SelfMutate) {
    selfMutate(bestlist_[0].first,bestlist_[1].first);
    bestlist_it++;
  }

  mapping_id = mapping_id_;

    // uint128_t mapping_index_factorization_id = mapping_id1[int(mapspace::Dimension::IndexFactorization)];  
    // uint128_t mapping_permutation_id = mapping_id2[int(mapspace::Dimension::LoopPermutation)];
    // uint128_t mapping_spatial_id = mapping_id2[int(mapspace::Dimension::Spatial)];
    // uint128_t mapping_datatype_bypass_id = mapping_id1[int(mapspace::Dimension::DatatypeBypass)];

  // std::cout << mapping_index_factorization_id << " ";
  // std::cout << mapping_permutation_id << " ";
  // std::cout << mapping_spatial_id << " ";
  // std::cout << mapping_datatype_bypass_id << std::endl;
  
  return true;
}

void GeneticSearch::Report(Status status, double cost)
{    

  if (status == Status::Success)
  {
    valid_mappings_++;
    std::cout << "Success" << std::endl;

    worklist_.push_back(std::make_pair(mapping_id_,cost));
  }
  else if (status == Status::MappingConstructionFailure)
    std::cout << "Mapping Failure" << std::endl;
  else if (status == Status::EvalFailure)
    std::cout << "Evaluation Failure" << std::endl;
  
  // Worklist filled, pick the best ones and start generating the next worklist
  if (state_ == State::Random && (int)(worklist_.size()) == worklist_max_size)
  {
    std::cout << "Worklist filled" << std::endl;

    // If this is the last iteration, terminate 
    iter_count++;
    if (iter_count >= MAX_ITER_COUNT) {
      state_ = State::Terminated;
      return;
    }

    // Sort the worklist
    worklist_.sort([](const std::pair<mapspace::ID, double> &x,
                      const std::pair<mapspace::ID, double> &y) 
    {
        return x.second < y.second;
    });

    // Copy the best ones to the best list
    bestlist_.clear();
    auto end = std::next(worklist_.begin(), bestlist_max_size);
    std::copy(worklist_.begin(), end, std::back_inserter(bestlist_));

    for (std::pair<mapspace::ID, double> p : bestlist_) {
      std::cout << p.first[0] << " " << p.second << std::endl;
    }

    // Clear the worklist
    worklist_.clear();

    // Switch to self mutation stage
    bestlist_it = bestlist_.begin();
    state_ = State::SelfMutate;
  }

  // Self mutations are done, switch to random stage
  if (state_== State::SelfMutate && bestlist_it == bestlist_.end()) {
    state_ = State::Random;
  }
}

} // namespace search
