// Dumps every split node of a saved decision forest as CSV (one row per node):
// tree,depth,node_id,parent_id,attribute,attribute_name,cond_type,threshold,
// num_examples,split_score. Used by the CC18 feature-selection study.
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, model, "", "Model directory.");
ABSL_FLAG(std::string, output, "", "Output CSV path ('' = stdout).");
ABSL_FLAG(bool, leaves, false, "Also emit leaf rows (attribute=-1).");

namespace ydf = yggdrasil_decision_forests;
using ydf::model::decision_tree::NodeWithChildren;
using ydf::model::decision_tree::proto::Condition;

namespace {

const char* CondName(const Condition& c) {
  switch (c.type_case()) {
    case Condition::kHigherCondition: return "higher";
    case Condition::kDiscretizedHigherCondition: return "disc_higher";
    case Condition::kContainsBitmapCondition: return "contains_bitmap";
    case Condition::kContainsCondition: return "contains";
    case Condition::kTrueValueCondition: return "true_value";
    case Condition::kNaCondition: return "na";
    case Condition::kObliqueCondition: return "oblique";
    default: return "other";
  }
}

void Walk(const NodeWithChildren& n, int tree, int depth, int parent_id,
          int* next_id, const ydf::dataset::proto::DataSpecification& spec,
          FILE* out, bool leaves) {
  const int id = (*next_id)++;
  if (n.IsLeaf()) {
    if (leaves) {
      fprintf(out, "%d,%d,%d,%d,-1,,leaf,nan,%lld,0\n", tree, depth, id,
              parent_id,
              (long long)n.node().num_pos_training_examples_without_weight());
    }
    return;
  }
  const auto& c = n.node().condition();
  const int attr = c.attribute();
  const auto& cond = c.condition();
  double thr = NAN;
  if (cond.has_higher_condition()) thr = cond.higher_condition().threshold();
  fprintf(out, "%d,%d,%d,%d,%d,%s,%s,%.9g,%lld,%.9g\n", tree, depth, id,
          parent_id, attr, spec.columns(attr).name().c_str(), CondName(cond),
          thr, (long long)c.num_training_examples_without_weight(),
          c.split_score());
  Walk(*n.neg_child(), tree, depth + 1, id, next_id, spec, out, leaves);
  Walk(*n.pos_child(), tree, depth + 1, id, next_id, spec, out, leaves);
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::unique_ptr<ydf::model::AbstractModel> model;
  CHECK_OK(ydf::model::LoadModel(absl::GetFlag(FLAGS_model), &model));
  const auto* forest =
      dynamic_cast<const ydf::model::DecisionForestInterface*>(model.get());
  CHECK(forest != nullptr) << "Model is not a decision forest";
  FILE* out = stdout;
  const std::string path = absl::GetFlag(FLAGS_output);
  if (!path.empty()) {
    out = fopen(path.c_str(), "w");
    CHECK(out != nullptr) << "Cannot open " << path;
  }
  fprintf(out,
          "tree,depth,node_id,parent_id,attribute,attribute_name,cond_type,"
          "threshold,num_examples,split_score\n");
  const auto& trees = forest->decision_trees();
  for (size_t t = 0; t < trees.size(); ++t) {
    int next_id = 0;
    Walk(trees[t]->root(), static_cast<int>(t), 0, -1, &next_id,
         model->data_spec(), out, absl::GetFlag(FLAGS_leaves));
  }
  if (out != stdout) fclose(out);
  return 0;
}
