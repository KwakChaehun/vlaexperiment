#include "tests/PIMBenchTestCases.h"
TEST_F(PIMBenchFixture, vla_query) {
  setPIMBenchTestCase(KernelType::GEMV, 4096, 1536);
  executeKernel();
  executePIMKernel();
}
TEST_F(PIMBenchFixture, vla_key) {
  setPIMBenchTestCase(KernelType::GEMV, 4096, 2048);
  executeKernel();
  executePIMKernel();
}
TEST_F(PIMBenchFixture, vla_ff_up) {
  setPIMBenchTestCase(KernelType::GEMV, 8192, 1536);
  executeKernel();
  executePIMKernel();
}
TEST_F(PIMBenchFixture, vla_ff_down) {
  setPIMBenchTestCase(KernelType::GEMV, 4096, 6144);
  executeKernel();
  executePIMKernel();
}
