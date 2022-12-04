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

#include <ncurses.h>

#include "applications/mapper/mapper-thread.hpp"

enum class Betterness
{
  Better,
  SlightlyBetter,
  SlightlyWorse,
  Worse
};

void MapperThread::run_tournament(std::vector<GeneticMapping> &progs,
                                  std::vector<int> &win_indices,
                                  const int n_progs,
                                  const int n_tours,
                                  const int tour_size,
                                  std::mt19937 &mt)
{
  win_indices.resize(n_tours);
  std::uniform_int_distribution<> rng(0, n_progs - 1);
  for (int idx = 0; idx < n_tours; ++idx)
  {

    int r = rng(mt);

    // Define optima values
    int opt = r % n_progs;
    float opt_score = progs[opt].cost;

    for (int s = 1; s < tour_size; ++s)
    {
      r = rng(mt);
      int curr = r % n_progs;
      float curr_score = progs[curr].cost;

      if (opt_score > curr_score)
      {
        opt = curr;
        opt_score = curr_score;
      }
    }

    win_indices[idx] = opt;
  }
}

static double Cost(const model::Topology::Stats &stats, const std::string metric)
{
  double cost;
  if (metric == "delay")
  {
    cost = static_cast<double>(stats.cycles);
  }
  else if (metric == "energy")
  {
    cost = stats.energy;
  }
  else if (metric == "last-level-accesses")
  {
    cost = stats.last_level_accesses;
  }
  else if (metric.compare(0, 9, "accesses-") == 0)
  {
    unsigned level = unsigned(atoi(metric.substr(9).c_str()));
    cost = stats.accesses.at(level);
  }
  else
  {
    assert(metric == "edp");
    cost = (stats.energy * stats.cycles);
  }
  return cost;
}

static Betterness IsBetterRecursive_(const model::Topology::Stats &candidate, const model::Topology::Stats &incumbent,
                                     const std::vector<std::string>::const_iterator metric,
                                     const std::vector<std::string>::const_iterator end)
{
  const double tolerance = 0.001;

  double candidate_cost = Cost(candidate, *metric);
  double incumbent_cost = Cost(incumbent, *metric);

  // Compute % improvement relative to incumbent. We need to
  // special-case cost == 0 to avoid a divide-by-zero error. Note that
  // cost == 0 is a legitimate cost for a mapping. Also note that lower
  // cost is better.
  double absolute_improvement = incumbent_cost - candidate_cost;
  double relative_improvement = incumbent_cost == 0 ? (candidate_cost == 0 ? 0 : absolute_improvement / candidate_cost) : absolute_improvement / incumbent_cost;

  if (fabs(relative_improvement) > tolerance)
  {
    // We have a clear winner.
    if (relative_improvement > 0)
      return Betterness::Better;
    else
      return Betterness::Worse;
  }
  else
  {
    // Within tolerance range, try to recurse.
    if (std::next(metric) == end)
    {
      // Base case. NOTE! Equality is categorized as SlightlyWorse (prefers incumbent).
      if (relative_improvement > 0)
        return Betterness::SlightlyBetter;
      else
        return Betterness::SlightlyWorse;
    }
    else
    {
      // Recursive call.
      Betterness lsm = IsBetterRecursive_(candidate, incumbent, std::next(metric), end);
      if (lsm == Betterness::Better || lsm == Betterness::Worse)
        return lsm;
      // NOTE! Equality is categorized as SlightlyWorse (prefers incumbent).
      else if (relative_improvement > 0)
        return Betterness::SlightlyBetter;
      else
        return Betterness::SlightlyWorse;
    }
  }
}

static inline bool IsBetter(const model::Topology::Stats &candidate, const model::Topology::Stats &incumbent,
                            const std::vector<std::string> &metrics)
{
  Betterness b = IsBetterRecursive_(candidate, incumbent, metrics.begin(), metrics.end());
  return (b == Betterness::Better || b == Betterness::SlightlyBetter);
}

bool EvaluationResult::UpdateIfBetter(const EvaluationResult &other, const std::vector<std::string> &metrics)
{
  bool updated = false;
  if (other.valid &&
      (!valid || IsBetter(other.stats, stats, metrics)))
  {
    valid = true;
    mapping = other.mapping;
    stats = other.stats;
    updated = true;
  }
  return updated;
}

//--------------------------------------------//
//              Failure Tracking              //
//--------------------------------------------//

std::map<FailClass, std::string> FailClassToString =
    {
        {FailClass::Fanout, "Fanout"},
        {FailClass::Capacity, "Capacity"}};

std::ostream &operator<<(std::ostream &out, const FailClass &fail_class)
{
  out << FailClassToString.at(fail_class);
  return out;
}

//--------------------------------------------//
//               Mapper Thread                //
//--------------------------------------------//

MapperThread::Stats::Stats() : distribution(0.0, 1.0)
{
}

void MapperThread::Stats::UpdateFails(FailClass fail_class, std::string fail_reason, unsigned level, const Mapping &mapping)
{
  // Find the data corresponding to this fail class.
  auto fail_bucket_it = fail_stats.find(fail_class);
  if (fail_bucket_it == fail_stats.end())
  {
    // We've never seen this fail class before.
    std::map<unsigned, FailInfo> fail_bucket;
    fail_bucket[level] = {.count = 1, .mapping = mapping, .reason = fail_reason};
    fail_stats[fail_class] = fail_bucket;
  }
  else
  {
    // We've seen this fail class, see if this level has
    // failed in this class.
    auto &fail_bucket = fail_bucket_it->second;
    auto fail_info_it = fail_bucket.find(level);
    if (fail_info_it == fail_bucket.end())
    {
      // No, this is the first time this level has failed in
      // this fail class, create a new entry.
      fail_bucket[level] = {.count = 1, .mapping = mapping, .reason = fail_reason};
    }
    else
    {
      // This level has already failed in this class,
      // increment its count.
      fail_info_it->second.count += 1;

      // p(x) = prob. that I switch to x when it arrives
      // p(0) = 1

      // P(x) = prob. that x is finally selected.
      // 1/N = P(0) = p(0).(1-p(1)).(1-p(2))...(1-p(N-1))
      // 1/N = P(1) =        (p(1)).(1-p(2))...(1-p(N-1))

      // p(x).(1-p(x+1)) = p(x+1)
      // ...
      // => p(x+1) = p(x) / [1+p(x)]
      // ...
      // => p(x) = 1/(1+x)

      // Compute the probability of switching (we've already computed count=x+1)
      double prob = 1 / fail_info_it->second.count.convert_to<double>();

      // Probabilistically update the mapping.
      double roll = distribution(generator);
      if (roll < prob)
      {
        fail_info_it->second.mapping = mapping;
        fail_info_it->second.reason = fail_reason;
      }
    }
  }
}

void MapperThread::loop_permute_mutation(const GeneticMapping &prog,
                                         GeneticMapping &p_out,
                                         std::mt19937 &rng,
                                         uint128_t &total_mappings,
                                         uint128_t &valid_mappings,
                                         uint128_t &invalid_mappings_mapcnstr,
                                         uint128_t &invalid_mappings_eval,
                                         model::Engine &engine)
{
  mapspace::ID mapping_id(mapspace_->AllSizes());
  mapping_id = prog.mapping_id;

    int iglobal;
  for (iglobal = 0; iglobal < 1000; ++iglobal)
  {
        std::cout << "loop permutation" << std::endl;

    search_->Next(mapping_id, 1);

    //
    // Begin Mapping. We do this in several stages with increasing algorithmic
    // complexity and attempt to bail out as quickly as possible at each stage.
    //
    bool success = true;

    // Stage 1: Construct a mapping from the mapping ID. This step can fail
    //          because the space of *legal* mappings isn't dense (unfortunately),
    //          so a mapping ID may point to an illegal mapping.
    Mapping mapping;

    auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
    success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                               [](bool cur, const mapspace::Status &status)
                               { return cur && status.success; });

    total_mappings++;

    if (!success)
    {
      search_->Report(search::Status::MappingConstructionFailure);
      continue;
    }

    // Stage 2: (Re)Configure a hardware model to evaluate the mapping
    //          on, and run some lightweight pre-checks that the
    //          model can use to quickly reject a nest.
    // engine.Spec(arch_specs_);
    auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
    success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus &status)
                               { return cur && status.success; });

    if (!success)
    {
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // Stage 3: Heavyweight evaluation.
    status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
    success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus &status)
                               { return cur && status.success; });
    if (!success)
    {
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // SUCCESS!!!
    auto stats = engine.GetTopology().GetStats();
    EvaluationResult result = {true, mapping, stats};

    valid_mappings++;
    if (log_stats_)
    {
      mutex_->lock();
      log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                  << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
      mutex_->unlock();
    }
    invalid_mappings_mapcnstr = 0;
    invalid_mappings_eval = 0;
    search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

    bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
    if (log_suboptimal_)
    {
      mutex_->lock();
      if (is_sparse_topology)
      {
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                    << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                    << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << std::endl;
      }
      else
      {
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                    << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << std::endl;
      }
      mutex_->unlock();
    }

    p_out.updateMapping(mapping_id, Cost(stats, optimization_metrics_.at(0)));
    break;
  }

  if (iglobal >= 1000)
  {

    while (true)
    {
      search_->Next(mapping_id,0);

      //
      // Begin Mapping. We do this in several stages with increasing algorithmic
      // complexity and attempt to bail out as quickly as possible at each stage.
      //
      bool success = true;

      // Stage 1: Construct a mapping from the mapping ID. This step can fail
      //          because the space of *legal* mappings isn't dense (unfortunately),
      //          so a mapping ID may point to an illegal mapping.
      Mapping mapping;

      auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
      success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                                 [](bool cur, const mapspace::Status &status)
                                 { return cur && status.success; });

      total_mappings++;

      if (!success)
      {
        search_->Report(search::Status::MappingConstructionFailure);
        continue;
      }

      // Stage 2: (Re)Configure a hardware model to evaluate the mapping
      //          on, and run some lightweight pre-checks that the
      //          model can use to quickly reject a nest.
      // engine.Spec(arch_specs_);
      auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
      success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus &status)
                                 { return cur && status.success; });

      if (!success)
      {
        search_->Report(search::Status::EvalFailure);
        continue;
      }

      // Stage 3: Heavyweight evaluation.
      status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
      success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus &status)
                                 { return cur && status.success; });
      if (!success)
      {
        search_->Report(search::Status::EvalFailure);
        continue;
      }

      // SUCCESS!!!
      auto stats = engine.GetTopology().GetStats();
      EvaluationResult result = {true, mapping, stats};

      valid_mappings++;
      if (log_stats_)
      {
        mutex_->lock();
        log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                    << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
        mutex_->unlock();
      }
      invalid_mappings_mapcnstr = 0;
      invalid_mappings_eval = 0;
      search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

      bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
      if (log_suboptimal_)
      {
        mutex_->lock();
        if (is_sparse_topology)
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                      << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                      << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                      << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                      << " | " << mapping.PrintCompact()
                      << std::endl;
        }
        else
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                      << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                      << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                      << " | " << mapping.PrintCompact()
                      << std::endl;
        }
        mutex_->unlock();
      }

      p_out.updateMapping(mapping_id, Cost(stats, optimization_metrics_.at(0)));
      break;
    }
  }
}
void MapperThread::data_bypass_mutation(const GeneticMapping &prog,
                                        GeneticMapping &p_out,
                                        std::mt19937 &rng,
                                        uint128_t &total_mappings,
                                        uint128_t &valid_mappings,
                                        uint128_t &invalid_mappings_mapcnstr,
                                        uint128_t &invalid_mappings_eval,
                                        model::Engine &engine)
{
  
  mapspace::ID mapping_id(mapspace_->AllSizes());
  mapping_id = prog.mapping_id;

    int iglobal;
  for (iglobal = 0; iglobal < 1000; ++iglobal)
  {
    std::cout << "dbypass" << std::endl;
    search_->Next(mapping_id, 2);

    //
    // Begin Mapping. We do this in several stages with increasing algorithmic
    // complexity and attempt to bail out as quickly as possible at each stage.
    //
    bool success = true;

    // Stage 1: Construct a mapping from the mapping ID. This step can fail
    //          because the space of *legal* mappings isn't dense (unfortunately),
    //          so a mapping ID may point to an illegal mapping.
    Mapping mapping;

    auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
    success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                               [](bool cur, const mapspace::Status &status)
                               { return cur && status.success; });

    total_mappings++;

    if (!success)
    {
      search_->Report(search::Status::MappingConstructionFailure);
      continue;
    }

    // Stage 2: (Re)Configure a hardware model to evaluate the mapping
    //          on, and run some lightweight pre-checks that the
    //          model can use to quickly reject a nest.
    // engine.Spec(arch_specs_);
    auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
    success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus &status)
                               { return cur && status.success; });

    if (!success)
    {
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // Stage 3: Heavyweight evaluation.
    status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
    success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus &status)
                               { return cur && status.success; });
    if (!success)
    {
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // SUCCESS!!!
    auto stats = engine.GetTopology().GetStats();
    EvaluationResult result = {true, mapping, stats};

    valid_mappings++;
    if (log_stats_)
    {
      mutex_->lock();
      log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                  << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
      mutex_->unlock();
    }
    invalid_mappings_mapcnstr = 0;
    invalid_mappings_eval = 0;
    search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

    bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
    if (log_suboptimal_)
    {
      mutex_->lock();
      if (is_sparse_topology)
      {
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                    << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                    << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << std::endl;
      }
      else
      {
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                    << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << std::endl;
      }
      mutex_->unlock();
    }

    p_out.updateMapping(mapping_id, Cost(stats, optimization_metrics_.at(0)));
    break;
  }

  if (iglobal >= 1000)
  {

    while (true)
    {
      search_->Next(mapping_id,0);

      //
      // Begin Mapping. We do this in several stages with increasing algorithmic
      // complexity and attempt to bail out as quickly as possible at each stage.
      //
      bool success = true;

      // Stage 1: Construct a mapping from the mapping ID. This step can fail
      //          because the space of *legal* mappings isn't dense (unfortunately),
      //          so a mapping ID may point to an illegal mapping.
      Mapping mapping;

      auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
      success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                                 [](bool cur, const mapspace::Status &status)
                                 { return cur && status.success; });

      total_mappings++;

      if (!success)
      {
        search_->Report(search::Status::MappingConstructionFailure);
        continue;
      }

      // Stage 2: (Re)Configure a hardware model to evaluate the mapping
      //          on, and run some lightweight pre-checks that the
      //          model can use to quickly reject a nest.
      // engine.Spec(arch_specs_);
      auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
      success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus &status)
                                 { return cur && status.success; });

      if (!success)
      {
        search_->Report(search::Status::EvalFailure);
        continue;
      }

      // Stage 3: Heavyweight evaluation.
      status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
      success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus &status)
                                 { return cur && status.success; });
      if (!success)
      {
        search_->Report(search::Status::EvalFailure);
        continue;
      }

      // SUCCESS!!!
      auto stats = engine.GetTopology().GetStats();
      EvaluationResult result = {true, mapping, stats};

      valid_mappings++;
      if (log_stats_)
      {
        mutex_->lock();
        log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                    << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
        mutex_->unlock();
      }
      invalid_mappings_mapcnstr = 0;
      invalid_mappings_eval = 0;
      search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

      bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
      if (log_suboptimal_)
      {
        mutex_->lock();
        if (is_sparse_topology)
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                      << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                      << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                      << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                      << " | " << mapping.PrintCompact()
                      << std::endl;
        }
        else
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                      << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                      << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                      << " | " << mapping.PrintCompact()
                      << std::endl;
        }
        mutex_->unlock();
      }

      p_out.updateMapping(mapping_id, Cost(stats, optimization_metrics_.at(0)));
      break;
    }
  }
}
void MapperThread::index_factorization_mutation(const GeneticMapping &prog,
                                                GeneticMapping &p_out,
                                                std::mt19937 &rng,
                                                uint128_t &total_mappings,
                                                uint128_t &valid_mappings,
                                                uint128_t &invalid_mappings_mapcnstr,
                                                uint128_t &invalid_mappings_eval,
                                                model::Engine &engine)
{

  mapspace::ID mapping_id(mapspace_->AllSizes());
  mapping_id = prog.mapping_id;

  int iglobal;
  for (iglobal = 0; iglobal < 1000; ++iglobal)
  {
        std::cout << "ifactorization" << std::endl;

    search_->Next(mapping_id, 3);

    //
    // Begin Mapping. We do this in several stages with increasing algorithmic
    // complexity and attempt to bail out as quickly as possible at each stage.
    //
    bool success = true;

    // Stage 1: Construct a mapping from the mapping ID. This step can fail
    //          because the space of *legal* mappings isn't dense (unfortunately),
    //          so a mapping ID may point to an illegal mapping.
    Mapping mapping;

    auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
    success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                               [](bool cur, const mapspace::Status &status)
                               { return cur && status.success; });

    total_mappings++;

    if (!success)
    {
      search_->Report(search::Status::MappingConstructionFailure);
      continue;
    }

    // Stage 2: (Re)Configure a hardware model to evaluate the mapping
    //          on, and run some lightweight pre-checks that the
    //          model can use to quickly reject a nest.
    // engine.Spec(arch_specs_);
    auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
    success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus &status)
                               { return cur && status.success; });

    if (!success)
    {
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // Stage 3: Heavyweight evaluation.
    status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
    success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus &status)
                               { return cur && status.success; });
    if (!success)
    {
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // SUCCESS!!!
    auto stats = engine.GetTopology().GetStats();
    EvaluationResult result = {true, mapping, stats};

    valid_mappings++;
    if (log_stats_)
    {
      mutex_->lock();
      log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                  << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
      mutex_->unlock();
    }
    invalid_mappings_mapcnstr = 0;
    invalid_mappings_eval = 0;
    search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

    bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
    if (log_suboptimal_)
    {
      mutex_->lock();
      if (is_sparse_topology)
      {
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                    << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                    << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << std::endl;
      }
      else
      {
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                    << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << std::endl;
      }
      mutex_->unlock();
    }

    p_out.updateMapping(mapping_id, Cost(stats, optimization_metrics_.at(0)));
    break;
  }

  if (iglobal >= 1000)
  {

    while (true)
    {
      search_->Next(mapping_id,0);

      //
      // Begin Mapping. We do this in several stages with increasing algorithmic
      // complexity and attempt to bail out as quickly as possible at each stage.
      //
      bool success = true;

      // Stage 1: Construct a mapping from the mapping ID. This step can fail
      //          because the space of *legal* mappings isn't dense (unfortunately),
      //          so a mapping ID may point to an illegal mapping.
      Mapping mapping;

      auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
      success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                                 [](bool cur, const mapspace::Status &status)
                                 { return cur && status.success; });

      total_mappings++;

      if (!success)
      {
        search_->Report(search::Status::MappingConstructionFailure);
        continue;
      }

      // Stage 2: (Re)Configure a hardware model to evaluate the mapping
      //          on, and run some lightweight pre-checks that the
      //          model can use to quickly reject a nest.
      // engine.Spec(arch_specs_);
      auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
      success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus &status)
                                 { return cur && status.success; });

      if (!success)
      {
        search_->Report(search::Status::EvalFailure);
        continue;
      }

      // Stage 3: Heavyweight evaluation.
      status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
      success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus &status)
                                 { return cur && status.success; });
      if (!success)
      {
        search_->Report(search::Status::EvalFailure);
        continue;
      }

      // SUCCESS!!!
      auto stats = engine.GetTopology().GetStats();
      EvaluationResult result = {true, mapping, stats};

      valid_mappings++;
      if (log_stats_)
      {
        mutex_->lock();
        log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                    << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
        mutex_->unlock();
      }
      invalid_mappings_mapcnstr = 0;
      invalid_mappings_eval = 0;
      search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

      bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
      if (log_suboptimal_)
      {
        mutex_->lock();
        if (is_sparse_topology)
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                      << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                      << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                      << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                      << " | " << mapping.PrintCompact()
                      << std::endl;
        }
        else
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                      << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                      << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                      << " | " << mapping.PrintCompact()
                      << std::endl;
        }
        mutex_->unlock();
      }

      p_out.updateMapping(mapping_id, Cost(stats, optimization_metrics_.at(0)));
      break;
    }
  }
}

void MapperThread::crossover(GeneticMapping &prog,
                             GeneticMapping &donor,
                             GeneticMapping &p_out,
                             std::mt19937 &rng,
                             uint128_t &total_mappings,
                             uint128_t &valid_mappings,
                             uint128_t &invalid_mappings_mapcnstr,
                             uint128_t &invalid_mappings_eval,
                             model::Engine &engine)
{
  std::uniform_int_distribution<> u01(0, 1);

  mapspace::ID mapping_id(mapspace_->AllSizes());
  std::vector<GeneticMapping> d;
  d.push_back(prog);
  d.push_back(donor);

  int iglobal;
  for (iglobal = 0; iglobal < 8; ++iglobal)
  {
    std::cout << "iglobal: " << iglobal << std::endl;
    int toss1 = u01(rng);
    int toss2 = u01(rng);
    int toss3 = u01(rng);

    mapping_id.Set(int(mapspace::Dimension::IndexFactorization), d[toss1].mapping_id[int(mapspace::Dimension::IndexFactorization)]);
    // std::cout << "iglobal12: " << iglobal << std::endl;
    mapping_id.Set(int(mapspace::Dimension::LoopPermutation), d[toss2].mapping_id[int(mapspace::Dimension::LoopPermutation)]);
    mapping_id.Set(int(mapspace::Dimension::Spatial), d[toss1].mapping_id[int(mapspace::Dimension::Spatial)]);
    mapping_id.Set(int(mapspace::Dimension::DatatypeBypass), d[toss3].mapping_id[int(mapspace::Dimension::DatatypeBypass)]);

    //
    // Begin Mapping. We do this in several stages with increasing algorithmic
    // complexity and attempt to bail out as quickly as possible at each stage.
    //
    bool success = true;

    // Stage 1: Construct a mapping from the mapping ID. This step can fail
    //          because the space of *legal* mappings isn't dense (unfortunately),
    //          so a mapping ID may point to an illegal mapping.
    Mapping mapping;

    auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
    success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                               [](bool cur, const mapspace::Status &status)
                               { return cur && status.success; });
    // std::cout << "iglobal2: " << iglobal << std::endl;
    total_mappings++;

    if (!success)
    {
      search_->Report(search::Status::MappingConstructionFailure);
      continue;
    }

    // Stage 2: (Re)Configure a hardware model to evaluate the mapping
    //          on, and run some lightweight pre-checks that the
    //          model can use to quickly reject a nest.
    // engine.Spec(arch_specs_);
    auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
    success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus &status)
                               { return cur && status.success; });

    if (!success)
    {
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // Stage 3: Heavyweight evaluation.
    status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
    success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus &status)
                               { return cur && status.success; });
    if (!success)
    {
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // SUCCESS!!!
    auto stats = engine.GetTopology().GetStats();
    EvaluationResult result = {true, mapping, stats};

    valid_mappings++;
    if (log_stats_)
    {
      mutex_->lock();
      log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                  << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
      mutex_->unlock();
    }
    invalid_mappings_mapcnstr = 0;
    invalid_mappings_eval = 0;
    search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

    bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
    if (log_suboptimal_)
    {
      mutex_->lock();
      if (is_sparse_topology)
      {
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                    << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                    << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << std::endl;
      }
      else
      {
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                    << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << std::endl;
      }
      mutex_->unlock();
    }

    p_out.updateMapping(mapping_id, Cost(stats, optimization_metrics_.at(0)));
    break;
  }

  if (iglobal >= 8)
  {

    while (true)
    {
      search_->Next(mapping_id,0);

      //
      // Begin Mapping. We do this in several stages with increasing algorithmic
      // complexity and attempt to bail out as quickly as possible at each stage.
      //
      bool success = true;

      // Stage 1: Construct a mapping from the mapping ID. This step can fail
      //          because the space of *legal* mappings isn't dense (unfortunately),
      //          so a mapping ID may point to an illegal mapping.
      Mapping mapping;

      auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
      success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                                 [](bool cur, const mapspace::Status &status)
                                 { return cur && status.success; });

      total_mappings++;

      if (!success)
      {
        search_->Report(search::Status::MappingConstructionFailure);
        continue;
      }

      // Stage 2: (Re)Configure a hardware model to evaluate the mapping
      //          on, and run some lightweight pre-checks that the
      //          model can use to quickly reject a nest.
      // engine.Spec(arch_specs_);
      auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
      success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus &status)
                                 { return cur && status.success; });

      if (!success)
      {
        search_->Report(search::Status::EvalFailure);
        continue;
      }

      // Stage 3: Heavyweight evaluation.
      status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
      success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus &status)
                                 { return cur && status.success; });
      if (!success)
      {
        search_->Report(search::Status::EvalFailure);
        continue;
      }

      // SUCCESS!!!
      auto stats = engine.GetTopology().GetStats();
      EvaluationResult result = {true, mapping, stats};

      valid_mappings++;
      if (log_stats_)
      {
        mutex_->lock();
        log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                    << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
        mutex_->unlock();
      }
      invalid_mappings_mapcnstr = 0;
      invalid_mappings_eval = 0;
      search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

      bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
      if (log_suboptimal_)
      {
        mutex_->lock();
        if (is_sparse_topology)
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                      << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                      << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                      << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                      << " | " << mapping.PrintCompact()
                      << std::endl;
        }
        else
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                      << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                      << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                      << " | " << mapping.PrintCompact()
                      << std::endl;
        }
        mutex_->unlock();
      }

      p_out.updateMapping(mapping_id, Cost(stats, optimization_metrics_.at(0)));
      break;
    }
  }
}

MapperThread::MapperThread(
    unsigned thread_id,
    search::SearchAlgorithm *search,
    mapspace::MapSpace *mapspace,
    std::mutex *mutex,
    uint128_t search_size,
    std::uint32_t timeout,
    std::uint32_t victory_condition,
    std::uint32_t nGenerations_,
    std::uint32_t population_size_,
    std::uint32_t tournament_size_,
    std::double_t p_crossover_,
    std::double_t p_loop_,
    std::double_t p_data_bypass_,
    std::double_t p_index_factorization_,
    std::double_t p_reproduce_,
    std::double_t p_random_,
    uint128_t sync_interval,
    bool log_stats,
    bool log_suboptimal,
    std::ostream &log_stream,
    bool live_status,
    bool diagnostics_on,
    bool penalize_consecutive_bypass_fails,
    std::vector<std::string> optimization_metrics,
    model::Engine::Specs arch_specs,
    problem::Workload &workload,
    sparse::SparseOptimizationInfo *sparse_optimizations,
    EvaluationResult *best) : thread_id_(thread_id),
                              search_(search),
                              mapspace_(mapspace),
                              mutex_(mutex),
                              search_size_(search_size),
                              timeout_(timeout),
                              victory_condition_(victory_condition),
                              nGenerations_(nGenerations_),
                              population_size_(population_size_),
                              tournament_size_(tournament_size_),
                              p_crossover_(p_crossover_),
                              p_loop_(p_loop_),
                              p_data_bypass_(p_data_bypass_),
                              p_index_factorization_(p_index_factorization_),
                              p_reproduce_(p_reproduce_),
                              p_random_(p_random_),
                              sync_interval_(sync_interval),
                              log_stats_(log_stats),
                              log_suboptimal_(log_suboptimal),
                              log_stream_(log_stream),
                              live_status_(live_status),
                              diagnostics_on_(diagnostics_on),
                              penalize_consecutive_bypass_fails_(penalize_consecutive_bypass_fails),
                              optimization_metrics_(optimization_metrics),
                              arch_specs_(arch_specs),
                              workload_(workload),
                              sparse_optimizations_(sparse_optimizations),
                              best_(best),
                              thread_(),
                              stats_()
{
}

void MapperThread::Start()
{
  // We can do this because std::thread is movable.
  thread_ = std::thread(&MapperThread::Run, this);
}

void MapperThread::Join()
{
  thread_.join();
}

const MapperThread::Stats &MapperThread::GetStats() const
{
  return stats_;
}

void MapperThread::Run()
{
  std::mt19937 mt;
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  next_worklist_.resize(population_size_);

  uint128_t total_mappings = 0;
  uint128_t valid_mappings = 0;
  uint128_t invalid_mappings_mapcnstr = 0;
  uint128_t invalid_mappings_eval = 0;
  std::uint32_t mappings_since_last_best_update = 0;

  const int ncurses_line_offset = 6;

  model::Engine engine;
  engine.Spec(arch_specs_);

  mapspace::ID prev_mapping_id;

  // =================
  // Main mapper loop.
  // =================

  if (search_->isGenetic)
  {
    uint32_t curgen = 0;
    while (curgen < nGenerations_)
    {
      curgen++;

      if (curgen == 1)
      {
        // Produce n random mappings

        // Fill
        while (current_worklist_.size() < population_size_)
        {
          mapspace::ID mapping_id;
          search_->Next(mapping_id,0);

          //
          // Begin Mapping. We do this in several stages with increasing algorithmic
          // complexity and attempt to bail out as quickly as possible at each stage.
          //
          bool success = true;

          // Stage 1: Construct a mapping from the mapping ID. This step can fail
          //          because the space of *legal* mappings isn't dense (unfortunately),
          //          so a mapping ID may point to an illegal mapping.
          Mapping mapping;

          auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
          success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                                     [](bool cur, const mapspace::Status &status)
                                     { return cur && status.success; });

          total_mappings++;

          if (!success)
          {
            search_->Report(search::Status::MappingConstructionFailure);
            continue;
          }

          // Stage 2: (Re)Configure a hardware model to evaluate the mapping
          //          on, and run some lightweight pre-checks that the
          //          model can use to quickly reject a nest.
          // engine.Spec(arch_specs_);
          auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
          success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                     [](bool cur, const model::EvalStatus &status)
                                     { return cur && status.success; });

          if (!success)
          {
            search_->Report(search::Status::EvalFailure);
            continue;
          }

          // Stage 3: Heavyweight evaluation.
          status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
          success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                     [](bool cur, const model::EvalStatus &status)
                                     { return cur && status.success; });
          if (!success)
          {
            search_->Report(search::Status::EvalFailure);
            continue;
          }

          // SUCCESS!!!
          auto stats = engine.GetTopology().GetStats();
          EvaluationResult result = {true, mapping, stats};

          valid_mappings++;
          if (log_stats_)
          {
            mutex_->lock();
            log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                        << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
            mutex_->unlock();
          }
          invalid_mappings_mapcnstr = 0;
          invalid_mappings_eval = 0;
          search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

          bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
          if (log_suboptimal_)
          {
            mutex_->lock();
            if (is_sparse_topology)
            {
              log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                          << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                          << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                          << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                          << " | " << mapping.PrintCompact()
                          << std::endl;
            }
            else
            {
              log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                          << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                          << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                          << " | " << mapping.PrintCompact()
                          << std::endl;
            }
            mutex_->unlock();
          }

          current_worklist_.push_back(GeneticMapping(mapping_id, Cost(stats, optimization_metrics_.at(0)), mutation_t::none));
        }

        // Print the mappings
        // std::cout << "Total mappings: " << total_mappings << "ieçüktieJütJÖJİEÜTJİEÜTİETktjöCÇJÖcçJÖCÇJÖ\n";
        // int cnt = 1;
        // for (auto mapping : current_worklist_)
        // {
        //   auto mf = mapping.mapping_id;
        //   uint128_t mapping_index_factorization_id = mf[int(mapspace::Dimension::IndexFactorization)];
        //   uint128_t mapping_permutation_id = mf[int(mapspace::Dimension::LoopPermutation)];
        //   uint128_t mapping_spatial_id = mf[int(mapspace::Dimension::Spatial)];
        //   uint128_t mapping_datatype_bypass_id = mf[int(mapspace::Dimension::DatatypeBypass)];

        //   std::cout << cnt++ << " ";
        //   std::cout << mapping_index_factorization_id << " ";
        //   std::cout << mapping_permutation_id << " ";
        //   std::cout << mapping_spatial_id << " ";
        //   std::cout << mapping_datatype_bypass_id << " ";
        //   std::cout << mapping.cost << std::endl;
        //   // Mapping mapping2;
        //   // auto construction_status = mapspace_->ConstructMapping(mapping.mapping_id, &mapping2, !diagnostics_on_);

        // }
      }
      else
      {
        // for(int j=0;j<current_worklist_.size();++j){
        //   mapspace::ID mapping_id100(mapspace_->AllSizes());
        //   mapping_id100.Set(int(mapspace::Dimension::IndexFactorization), current_worklist_[j].mapping_id[int(mapspace::Dimension::IndexFactorization)]);
        // }
        double mut_probs[6];
        mut_probs[0] = p_crossover_;
        mut_probs[1] = p_loop_;
        mut_probs[2] = p_data_bypass_;
        mut_probs[3] = p_index_factorization_;
        mut_probs[4] = p_reproduce_;
        mut_probs[5] = p_random_;
        std::partial_sum(mut_probs, mut_probs + 6, mut_probs);

        uint32_t n_tours = population_size_;
        int nc = 0;
        int nr = 0;
        for (uint32_t i = 0; i < population_size_; ++i)
        {
          double prob = u01(mt);

          if (prob < mut_probs[0])
          {
            next_worklist_[i].mut_type = mutation_t::crossover;
            n_tours++;
            nc++;
          }
          else if (prob < mut_probs[1])
          {
            next_worklist_[i].mut_type = mutation_t::loop;
          }
          else if (prob < mut_probs[2])
          {
            next_worklist_[i].mut_type = mutation_t::data_bypass;
          }
          else if (prob < mut_probs[3])
          {
            next_worklist_[i].mut_type = mutation_t::index_factorization;
          }
          else if (prob < mut_probs[4])
          {
            next_worklist_[i].mut_type = mutation_t::reproduce;
          }
          else
          {
            next_worklist_[i].mut_type = mutation_t::random;
            n_tours--;
            nr++;
          }
        }

        // for (uint32_t i = 0; i < population_size_; ++i) {
        //   if (next_worklist_[i].mut_type == mutation_t::crossover)
        //     std::cout << i << " Crossover  !" << nc<<"!\n" ;
        //   else if (next_worklist_[i].mut_type == mutation_t::loop)
        //     std::cout << i << " loop\n";
        //   else if (next_worklist_[i].mut_type == mutation_t::data_bypass)
        //     std::cout << i << " data_bypass\n";
        //   else if (next_worklist_[i].mut_type == mutation_t::index_factorization)
        //     std::cout << i << " index_factorization\n";
        //   else if (next_worklist_[i].mut_type == mutation_t::reproduce)
        //     std::cout << i << " reproduce\n";
        //     else if (next_worklist_[i].mut_type == mutation_t::random)
        //     std::cout << i << " random !" << nr <<"!\n";
        // }
        // std::cout << " nc=" << nc << " , nr=" << nr <<"\n";
        // std::cout << "No of tournaments: " << n_tours << "\n";
        // std::cout << "Population: " << population_size_ << "\n";

        std::vector<int> win_indices;
        run_tournament(current_worklist_, win_indices, population_size_,
                       n_tours, tournament_size_, mt);
        std::cout << "finished tournaments"
                  << std::endl;
        // std::cout << win_indices.size() << std::endl;
        // for (int i = 0; i < (int)win_indices.size(); ++i)
        // {
        //   if ((uint32_t)win_indices[i] < population_size_)
        //     // std::cout << "yes" << i << "\n";
        //   else
        //     // std::cout << "no" << i << "\n";
        // }

        int pos = 0;
        int donor_pos = 0;
        int idx = 0;
        while (pos < n_tours)
        { std::cout << "Position" << pos << std::endl;
          if (next_worklist_[idx].mut_type == mutation_t::random)
          {
            std::cout << "random search" << std::endl;

            while (true)
            {
              std::cout << "random search" << std::endl;
              mapspace::ID mapping_id;
              search_->Next(mapping_id,0);

              //
              // Begin Mapping. We do this in several stages with increasing algorithmic
              // complexity and attempt to bail out as quickly as possible at each stage.
              //
              bool success = true;

              // Stage 1: Construct a mapping from the mapping ID. This step can fail
              //          because the space of *legal* mappings isn't dense (unfortunately),
              //          so a mapping ID may point to an illegal mapping.
              Mapping mapping;

              auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
              success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                                         [](bool cur, const mapspace::Status &status)
                                         { return cur && status.success; });

              total_mappings++;

              if (!success)
              {
                search_->Report(search::Status::MappingConstructionFailure);
                continue;
              }

              // Stage 2: (Re)Configure a hardware model to evaluate the mapping
              //          on, and run some lightweight pre-checks that the
              //          model can use to quickly reject a nest.
              // engine.Spec(arch_specs_);
              auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
              success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                         [](bool cur, const model::EvalStatus &status)
                                         { return cur && status.success; });

              if (!success)
              {
                search_->Report(search::Status::EvalFailure);
                continue;
              }

              // Stage 3: Heavyweight evaluation.
              status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
              success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                         [](bool cur, const model::EvalStatus &status)
                                         { return cur && status.success; });
              if (!success)
              {
                search_->Report(search::Status::EvalFailure);
                continue;
              }

              // SUCCESS!!!
              auto stats = engine.GetTopology().GetStats();
              EvaluationResult result = {true, mapping, stats};

              valid_mappings++;
              if (log_stats_)
              {
                mutex_->lock();
                log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                            << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
                mutex_->unlock();
              }
              invalid_mappings_mapcnstr = 0;
              invalid_mappings_eval = 0;
              search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

              bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
              if (log_suboptimal_)
              {
                mutex_->lock();
                if (is_sparse_topology)
                {
                  log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                              << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                              << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                              << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                              << " | " << mapping.PrintCompact()
                              << std::endl;
                }
                else
                {
                  log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                              << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                              << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                              << " | " << mapping.PrintCompact()
                              << std::endl;
                }
                mutex_->unlock();
              }

              next_worklist_[idx].updateMapping(mapping_id, Cost(stats, optimization_metrics_.at(0)));
              break;
            }
          }
          else
          {
            auto parent_index = win_indices[pos++];

            if (next_worklist_[idx].mut_type == mutation_t::crossover)
            {
              std::cout << "crossover" << std::endl;

              // Get secondary index
              auto donor_index = win_indices[pos++];
              double c1 = next_worklist_[idx].cost;
              crossover(
                  current_worklist_[parent_index], current_worklist_[donor_index], next_worklist_[idx], mt,
                  total_mappings,
                  valid_mappings,
                  invalid_mappings_mapcnstr,
                  invalid_mappings_eval,
                  engine);
              double c2 = next_worklist_[idx].cost;
              // if (c2 != c1)
              //   std::cout << idx << " yes\n";
              // else
              //   std::cout << idx << " no\n";
            }
            else if (next_worklist_[idx].mut_type == mutation_t::loop)
            {
                            std::cout << "loop permute" << std::endl;

              loop_permute_mutation(current_worklist_[parent_index], next_worklist_[idx], mt,
                                    total_mappings,
                                    valid_mappings,
                                    invalid_mappings_mapcnstr,
                                    invalid_mappings_eval,
                                    engine);
            }
            else if (next_worklist_[idx].mut_type == mutation_t::data_bypass)
            {
                            std::cout << "dbypass" << std::endl;

              data_bypass_mutation(current_worklist_[parent_index], next_worklist_[idx], mt,
                                   total_mappings,
                                   valid_mappings,
                                   invalid_mappings_mapcnstr,
                                   invalid_mappings_eval,
                                   engine);
            }
            else if (next_worklist_[idx].mut_type == mutation_t::index_factorization)
            {
                            std::cout << "index factorization" << std::endl;

              index_factorization_mutation(current_worklist_[parent_index], next_worklist_[idx], mt,
                                           total_mappings,
                                           valid_mappings,
                                           invalid_mappings_mapcnstr,
                                           invalid_mappings_eval,
                                           engine);
            }
            else if (next_worklist_[idx].mut_type == mutation_t::reproduce)
            {
                            std::cout << "reproduce" << std::endl;

              next_worklist_[idx] = current_worklist_[parent_index];
            }
            else
            {
                            std::cout << "no way" << std::endl;

              // Should not come here
            }
          }
          ++idx;
        }
        std::cout << "generation " << curgen << " done" << std::endl;
      }
      //UPDATE THE WORKLIST FOR THE NEXT GENERATION
      current_worklist_ = next_worklist_;
    } // while ()
  }
  else
  {
    while (true)
    {
      if (live_status_)
      {
        std::stringstream msg;

        msg << std::setw(3) << thread_id_ << std::setw(11) << total_mappings
            << std::setw(11) << (total_mappings - valid_mappings) << std::setw(11) << valid_mappings
            << std::setw(11) << invalid_mappings_mapcnstr + invalid_mappings_eval
            << std::setw(11) << mappings_since_last_best_update;

        if (valid_mappings > 0)
        {
          msg << std::setw(10) << OUT_FLOAT_FORMAT << std::setprecision(2) << (stats_.thread_best.stats.utilization * 100) << "%"
              << std::setw(11) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats_.thread_best.stats.energy / stats_.thread_best.stats.algorithmic_computes;
        }

        mutex_->lock();
        mvaddstr(thread_id_ + ncurses_line_offset, 0, msg.str().c_str());
        refresh();
        mutex_->unlock();
      }

      // Termination conditions.
      bool terminate = false;

      if (gTerminate)
      {
        mutex_->lock();
        log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: "
                    << "global termination flag activated, terminating search."
                    << std::endl;
        mutex_->unlock();
        terminate = true;
      }

      if (search_size_ > 0 && valid_mappings == search_size_)
      {
        mutex_->lock();
        log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << search_size_
                    << " valid mappings found, terminating search."
                    << std::endl;
        mutex_->unlock();
        terminate = true;
      }

      if (victory_condition_ > 0 && mappings_since_last_best_update == victory_condition_)
      {
        mutex_->lock();
        log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << victory_condition_
                    << " suboptimal mappings found since the last upgrade, terminating search."
                    << std::endl;
        mutex_->unlock();
        terminate = true;
      }

      if ((invalid_mappings_mapcnstr + invalid_mappings_eval) > 0 &&
          (invalid_mappings_mapcnstr + invalid_mappings_eval) == timeout_)
      {
        mutex_->lock();
        log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << timeout_
                    << " invalid mappings (" << invalid_mappings_mapcnstr << " fanout, "
                    << invalid_mappings_eval << " capacity) found since the last valid mapping, "
                    << "terminating search." << std::endl;
        mutex_->unlock();
        terminate = true;
      }

      // Try to obtain the next mapping from the search algorithm.
      mapspace::ID mapping_id;
      if (!search_->Next(mapping_id))
      {
        mutex_->lock();
        log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: "
                    << "search algorithm is done, terminating search."
                    << std::endl;
        mutex_->unlock();
        terminate = true;
      }

      // Terminate.
      if (terminate)
      {
        if (live_status_)
        {
          mutex_->lock();
          mvaddstr(thread_id_ + ncurses_line_offset, 0, "-");
          refresh();
          mutex_->unlock();
        }
        break;
      }

      //
      // Periodically sync thread_best with global best.
      //
      if (total_mappings != 0 && sync_interval_ > 0 && total_mappings % sync_interval_ == 0)
      {
        mutex_->lock();

        // Sync from global best to thread_best.
        bool global_pulled = false;
        if (best_->valid)
        {
          if (stats_.thread_best.UpdateIfBetter(*best_, optimization_metrics_))
          {
            global_pulled = true;
          }
        }

        // Sync from thread_best to global best.
        if (stats_.thread_best.valid && !global_pulled)
        {
          best_->UpdateIfBetter(stats_.thread_best, optimization_metrics_);
        }

        mutex_->unlock();
      }

      //
      // Check if the only change vs. the previous mapping was in the Bypass
      // dimension. This is useful later.
      //
      bool only_bypass_changed = false;
      if (total_mappings > 1)
      {
        bool match = true;
        for (unsigned idim = 0; idim < unsigned(mapspace::Dimension::Num); idim++)
        {
          if (mapspace::Dimension(idim) != mapspace::Dimension::DatatypeBypass)
            match &= (mapping_id[idim] == prev_mapping_id[idim]);
        }
        only_bypass_changed = match;
      }
      prev_mapping_id = mapping_id;

      //
      // Begin Mapping. We do this in several stages with increasing algorithmic
      // complexity and attempt to bail out as quickly as possible at each stage.
      //
      bool success = true;

      // Stage 1: Construct a mapping from the mapping ID. This step can fail
      //          because the space of *legal* mappings isn't dense (unfortunately),
      //          so a mapping ID may point to an illegal mapping.
      Mapping mapping;

      auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
      success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                                 [](bool cur, const mapspace::Status &status)
                                 { return cur && status.success; });

      total_mappings++;

      if (!success)
      {
        invalid_mappings_mapcnstr++;
        if (diagnostics_on_)
        {
          for (unsigned level = 0; level < construction_status.size(); level++)
            if (!construction_status.at(level).success)
              stats_.UpdateFails(FailClass::Fanout, construction_status.at(level).fail_reason, level, mapping);
        }
        search_->Report(search::Status::MappingConstructionFailure);
        continue;
      }

      // Stage 2: (Re)Configure a hardware model to evaluate the mapping
      //          on, and run some lightweight pre-checks that the
      //          model can use to quickly reject a nest.
      // engine.Spec(arch_specs_);
      auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
      success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus &status)
                                 { return cur && status.success; });

      if (!success)
      {
        // Pre-evaluation failed.
        // If the only change in this mapping vs. the previous mapping was in
        // its dataspace bypass scheme, then we may not want to make this
        // failure count towards the timeout termination trigger.
        if (penalize_consecutive_bypass_fails_ || !only_bypass_changed)
        {
          invalid_mappings_eval++;
        }

        if (diagnostics_on_)
        {
          for (unsigned level = 0; level < status_per_level.size(); level++)
            if (!status_per_level.at(level).success)
              stats_.UpdateFails(FailClass::Capacity, status_per_level.at(level).fail_reason, level, mapping);
        }
        search_->Report(search::Status::EvalFailure);
        continue;
      }

      // Stage 3: Heavyweight evaluation.
      status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
      success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                                 [](bool cur, const model::EvalStatus &status)
                                 { return cur && status.success; });
      if (!success)
      {
        // Evaluation failed.
        // If the only change in this mapping vs. the previous mapping was in
        // its dataspace bypass scheme, then we may not want to make this
        // failure count towards the timeout termination trigger.
        if (penalize_consecutive_bypass_fails_ || !only_bypass_changed)
        {
          invalid_mappings_eval++;
        }

        if (diagnostics_on_)
        {
          for (unsigned level = 0; level < status_per_level.size(); level++)
            if (!status_per_level.at(level).success)
              stats_.UpdateFails(FailClass::Capacity, status_per_level.at(level).fail_reason, level, mapping);
        }
        search_->Report(search::Status::EvalFailure);
        continue;
      }

      // SUCCESS!!!
      auto stats = engine.GetTopology().GetStats();
      EvaluationResult result = {true, mapping, stats};

      valid_mappings++;
      if (log_stats_)
      {
        mutex_->lock();
        log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                    << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
        mutex_->unlock();
      }
      invalid_mappings_mapcnstr = 0;
      invalid_mappings_eval = 0;
      search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

      bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
      if (log_suboptimal_)
      {
        mutex_->lock();
        if (is_sparse_topology)
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                      << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                      << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                      << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                      << " | " << mapping.PrintCompact()
                      << std::endl;
        }
        else
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                      << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                      << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                      << " | " << mapping.PrintCompact()
                      << std::endl;
        }
        mutex_->unlock();
      }

      // Is the new mapping "better" than the previous best mapping?
      if (stats_.thread_best.UpdateIfBetter(result, optimization_metrics_))
      {
        if (log_stats_)
        {
          // FIXME: improvement only captures the primary stat.
          double improvement = stats_.thread_best.valid ? (Cost(stats_.thread_best.stats, optimization_metrics_.at(0)) - Cost(stats, optimization_metrics_.at(0))) /
                                                              Cost(stats_.thread_best.stats, optimization_metrics_.at(0))
                                                        : 1.0;
          mutex_->lock();
          log_stream_ << "[" << thread_id_ << "] UPDATE " << total_mappings << " " << valid_mappings
                      << " " << mappings_since_last_best_update << " " << improvement << std::endl;
          mutex_->unlock();
        }

        if (!log_suboptimal_)
        {
          mutex_->lock();
          if (is_sparse_topology)
          {
            log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                        << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                        << " | pJ/Algorithmic-Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                        << " | pJ/Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                        << " | " << mapping.PrintCompact()
                        << std::endl;
          }
          else
          {
            log_stream_ << "[" << std::setw(3) << thread_id_ << "]"
                        << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization
                        << " | pJ/Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                        << " | " << mapping.PrintCompact()
                        << std::endl;
          }
          mutex_->unlock();
        }

        mappings_since_last_best_update = 0;
      }
      else
      {
        // If the only change in this mapping vs. the previous mapping was in
        // its dataspace bypass scheme, then we may not want to make this
        // failure count towards the timeout termination trigger.
        if (penalize_consecutive_bypass_fails_ || !only_bypass_changed)
        {
          mappings_since_last_best_update++;
        }
      }
    } // while ()
  }
  int iglobal;
  for(iglobal=0;iglobal<current_worklist_.size();++iglobal){
    if(current_worklist_[iglobal].cost != next_worklist_[iglobal].cost)
    {
      std::cout << "you fucked up" << std::endl;
    }
  }
  if(iglobal >= current_worklist_.size())
  {
    std::cout << "all good" << std::endl;
  }
}
