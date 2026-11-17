
#include "tests/PIMBenchTestCases.h"
#include <cstdlib>
#include <cmath>

class VLABatchProbe : public GemvPIMBenchTest {
 public:
  VLABatchProbe(unsigned b, unsigned o, unsigned i)
      : GemvPIMBenchTest(KernelType::GEMV,b,o,i) {
    // The upstream dummy fixture allocates only one input vector.
    BurstType zero; zero.set(0.0f);
    dim_data_->input_npbst_.bData.resize(b*dim_data_->input_npbst_.bShape[1],zero);
  }
  void functional() {
    const unsigned stride=in_/16;
    BurstType zero; zero.set(0.0f);
    auto& w=dim_data_->weight_npbst_.bData;
    w.resize(out_*stride,zero);
    // Exercise every FP16 lane and both bank parities, with distinct output
    // tiles and batches; exact binary fractions avoid tolerance ambiguity.
    for(unsigned r=0;r<out_;++r) {
      for(unsigned lane=0;lane<16;++lane) {
        w[r*stride].fp16Data_[lane]=fp16((r%7+1)/16.0f);
        w[r*stride+8].fp16Data_[lane]=fp16((r/4096+1)/16.0f);
      }
    }
    auto& x=dim_data_->input_npbst_.bData;
    for(unsigned b=0;b<batch_;++b) {
      x[b*stride].set(fp16(float(b+1))); x[b*stride+8].set(fp16(float(b+2)));
    }
    kernel_->preloadGemv(&dim_data_->weight_npbst_);
    kernel_->executeGemv(&dim_data_->weight_npbst_,&dim_data_->input_npbst_,false);
    auto result=readback(); kernel_->runPIM();
    for(unsigned b=0;b<batch_;++b) for(unsigned r=0;r<out_;++r) {
      float actual=float(result[b*out_+r].fp16ReduceSum());
      float expected=(r%7+1)*(b+1)+(r/4096+1)*(b+2);
      ASSERT_FLOAT_EQ(actual,expected) << "batch=" << b << " row=" << r;
    }
    std::cout << "VLA_FUNCTIONAL " << batch_*out_ << std::endl;
  }
  std::vector<BurstType> readback() {
    std::vector<BurstType> result(batch_*out_);
    unsigned col=kernel_->getResultColGemv(in_/16,out_);
    kernel_->readResult(result.data(),pimBankType::ODD_BANK,batch_*out_,0,0,col);
    return result;
  }
  void timing() {
    uint64_t compute=measureCycle(true);
    auto result=readback(); kernel_->runPIM();
    uint64_t total=kernel_->getCycle();
    std::cout << "VLA_CYCLES " << batch_ << " " << out_ << " " << in_
              << " " << compute << " " << total-compute << std::endl;
  }
};

TEST(VLAH50, functional_batch_tiles) { VLABatchProbe p(3,8192,256); p.functional(); }
TEST(VLAH50, timing) {
  VLABatchProbe p(std::atoi(std::getenv("VLA_VECTORS")),
                  std::atoi(std::getenv("VLA_OUT")),std::atoi(std::getenv("VLA_IN")));
  p.timing();
}
