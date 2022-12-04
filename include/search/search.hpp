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

#include "mapspaces/mapspace-base.hpp"

namespace search
{

enum class Status
{
  Success,
  MappingConstructionFailure,
  EvalFailure
};

class SearchAlgorithm
{ 
 public:
 std::array<PatternGenerator128*, int(mapspace::Dimension::Num)> pgens_;
  bool isGenetic = false;
  SearchAlgorithm() {}
  virtual ~SearchAlgorithm() {}
  virtual bool Next(mapspace::ID& mapping_id) = 0;
  virtual bool Next(mapspace::ID& mapping_id, int mode) { return false; }
  virtual void Report(Status status, double cost = 0) = 0;
  // void run_tournament(std::vector<GeneticMapping>& progs,
  //                   std::vector<int>& win_indices,
  //                                         const int n_progs,
  //                                         const int n_tours,
  //                                         const int tour_size,
  //                                         std::mt19937& mt) {}
};

} // namespace search

