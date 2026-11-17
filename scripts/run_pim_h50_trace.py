"""Simulate the full token batches of dense projections from a native H50 trace.

This is a serialized GEMV mapping, NOT an optimized token-parallel PIM GEMM.
No attention/activation is executed on PIM. Readback contains 16 FP16 partial
sums per padded output, which must be reduced externally. Model weights are
resident and excluded from timing. A synthetic multi-tile functional test
checks batching/addressing before dummy-data timing probes are accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

DRIVER = r'''
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
'''


def main():
    import os
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Use new output')
    sim=ROOT/'.runtime/PIMSimulator'
    # Minimal correction: upstream j+b aliases batches across output tiles.
    # Keep upstream source local; record the exact edit and hash in the manifest.
    target=sim/'src/tests/PIMKernel.cpp'; old='(j + b) * num_grfB_'; new='(b * num_output_tiles + j) * num_grfB_'
    src=target.read_text()
    if old in src:
        assert src.count(old)==1
        target.write_text(src.replace(old,new))
    elif new not in src:raise ValueError('Unsupported upstream source')
    (sim/'src/tests/VLAH50Probe.cpp').write_text(DRIVER)
    (out/'VLAH50Probe.cpp').write_text(DRIVER)
    with (out/'build.log').open('w') as f:
        subprocess.run([str(ROOT/'.runtime/pim-tools/bin/scons'),'NO_LIBRARY=1'],cwd=sim,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (out/'functional.log').open('w') as f:
        subprocess.run(['./sim','--gtest_filter=VLAH50.functional_batch_tiles:PIMKernelFixture.gemv'],cwd=sim,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
    functional=(out/'functional.log').read_text()
    assert 'VLA_FUNCTIONAL 24576' in functional and '[  PASSED  ] 2 tests.' in functional
    traces=json.loads(Path(args.trace).read_text())
    shapes=sorted({(r['vectors'],((r['output_dim']+4095)//4096)*4096,((r['input_dim']+255)//256)*256)
                   for tr in traces for r in tr['operators']})
    cases=[]
    for vectors,output_dim,input_dim in shapes:
        name=f'v{vectors}_o{output_dim}_i{input_dim}'
        log=out/(name+'.log')
        with log.open('w') as f:
            subprocess.run(['./sim','--gtest_filter=VLAH50.timing'],cwd=sim,stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,VLA_VECTORS=str(vectors),VLA_OUT=str(output_dim),VLA_IN=str(input_dim)),check=True,timeout=600)
        match=re.search(r'VLA_CYCLES (\d+) (\d+) (\d+) (\d+) (\d+)',log.read_text())
        assert match
        v,o,i,compute,readback=map(int,match.groups()); assert (v,o,i)==(vectors,output_dim,input_dim)
        row=dict(vectors=v,padded_output_dim=o,padded_input_dim=i,compute_cycles=compute,
                 readback_cycles=readback,compute_ms=compute/1e6,readback_ms=readback/1e6,
                 partial_output_bytes=v*o*16*2,padded_input_bytes=v*i*2)
        cases.append(row)
        with (out/'shapes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(event='pim_batch',**row)),flush=True)
    result=dict(cases=cases,tck_ns=1,channels=64,blocks_per_channel=8,
        upstream_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=sim,text=True).strip(),
        local_kernel_edit=dict(file='src/tests/PIMKernel.cpp',old=old,new=new,sha256=hashlib.sha256(target.read_bytes()).hexdigest()),
        dummy_batch_allocation='Driver resizes input burst array to vectors * padded_input_dim / 16',
        functional_outputs=24576,reference_single_vector_test_passed=True,
        trace_sha256=hashlib.sha256(Path(args.trace).read_bytes()).hexdigest(),
        scope=__doc__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
