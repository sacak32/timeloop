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
    state_(State::Ready),
    mapping_id_(mapspace->AllSizes()),
    masking_space_covered_(mapspace_->Size(mapspace::Dimension::DatatypeBypass)),
    valid_mappings_(0)
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

  generateInitialMappings();
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

void GeneticSearch::generateInitialMappings()
{
  for (int i = 0; i < worklist_size; i++)
  {
    Roll(mapspace::Dimension::IndexFactorization);
    Roll(mapspace::Dimension::LoopPermutation);
    Roll(mapspace::Dimension::Spatial);
    Roll(mapspace::Dimension::DatatypeBypass);
    worklist_.push_back(mapping_id_);
  }
}

bool GeneticSearch::Next(mapspace::ID& mapping_id)
{
  if (state_ == State::Terminated)
  {
    return false;
  }

  assert(state_ == State::Ready);
    
  state_ = State::WaitingForStatus;
    
  mapping_id = worklist_.front();
  worklist.pop_front();
  return true;
}

void GeneticSearch::Report(Status status, double cost)
{
  (void) cost;
    
  assert(state_ == State::WaitingForStatus);

  if (status == Status::Success)
  {
    valid_mappings_++;
    std::cout << "Success" << std::endl;
  }
  else if (status == Status::MappingConstructionFailure)
    std::cout << "Mapping Failure" << std::endl;
  else if (status == Status::EvalFailure)
    std::cout << "Evaluation Failure" << std::endl;
    
  if (worklist_.empty())
  {
    state_ = State::Terminated;
  }
  else
  {
    state_ = State::Ready;
  }
}

} // namespace search
