/**************************************************************************//**
*
* INTEL CONFIDENTIAL
* Copyright 2023
* Intel Corporation All Rights Reserved.
*
* The source code contained or described herein and all documents related to the
* source code ("Material") are owned by Intel Corporation or its suppliers or
* licensors. Title to the Material remains with Intel Corporation or its suppliers
* and licensors. The Material contains trade secrets and proprietary and confidential
* information of Intel or its suppliers and licensors. The Material is protected by
* worldwide copyright and trade secret laws and treaty provisions. No part of the
* Material may be used, copied, reproduced, modified, published, uploaded, posted
* transmitted, distributed, or disclosed in any way without Intel's prior express
* written permission.
*
* No license under any patent, copyright, trade secret or other intellectual
* property right is granted to or conferred upon you by disclosure or delivery
* of the Materials, either expressly, by implication, inducement, estoppel
* or otherwise. Any license under such intellectual property rights must be
* express and approved by Intel in writing.
*
* File Name:
*
* Abstract:
*
* Notes:
*
\*****************************************************************************/

#include <CLI/CLI.hpp>
#include "gtest/gtest.h"

#include "config.h"

int main( int argc, char* argv[ ] )
{
    CLI::App tests_app{ "App to tests dml kernels.", "Tester." };
    tests_app.allow_extras(true);
    tests_app.add_flag("--run_dml", g_test_config.run_dml, "Run DirectML dispatcher instead of POC umd d3d12 dispatcher class.")->default_val(false);
    tests_app.add_option("--iters", g_test_config.iterations, "How many iterations to run given dispatcher. This record to single command list and executes after cmd list is recorded.")->default_val(1);
    try {
        tests_app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e) {
        return tests_app.exit(e);
    }


    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}