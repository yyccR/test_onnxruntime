
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cargs.h"
#include "sherpa-onnx/c-api/c-api.h"

static struct cag_option options[] = {
        {.identifier = 'h',
                .access_letters = "h",
                .access_name = "help",
                .description = "Show help"},
        {.access_name = "vits-model",
                .value_name = "/path/to/xxx.onnx",
                .identifier = '0',
                .description = "Path to VITS model"},
        {.access_name = "vits-lexicon",
                .value_name = "/path/to/lexicon.txt",
                .identifier = '1',
                .description = "Path to lexicon.txt for VITS models"},
        {.access_name = "vits-tokens",
                .value_name = "/path/to/tokens.txt",
                .identifier = '2',
                .description = "Path to tokens.txt for VITS models"},
        {.access_name = "vits-noise-scale",
                .value_name = "0.667",
                .identifier = '3',
                .description = "noise_scale for VITS models"},
        {.access_name = "vits-noise-scale-w",
                .value_name = "0.8",
                .identifier = '4',
                .description = "noise_scale_w for VITS models"},
        {.access_name = "vits-length-scale",
                .value_name = "1.0",
                .identifier = '5',
                .description =
                "length_scale for VITS models. Default to 1. You can tune it "
                "to change the speech speed. small -> faster; large -> slower. "},
        {.access_name = "num-threads",
                .value_name = "1",
                .identifier = '6',
                .description = "Number of threads"},
        {.access_name = "provider",
                .value_name = "cpu",
                .identifier = '7',
                .description = "Provider: cpu (default), cuda, coreml"},
        {.access_name = "debug",
                .value_name = "0",
                .identifier = '8',
                .description = "1 to show debug messages while loading the model"},
        {.access_name = "sid",
                .value_name = "0",
                .identifier = '9',
                .description = "Speaker ID. Default to 0. Note it is not used for "
                               "single-speaker models."},
        {.access_name = "output-filename",
                .value_name = "./generated.wav",
                .identifier = 'a',
                .description =
                "Filename to save the generated audio. Default to ./generated.wav"},

        {.access_name = "tts-rule-fsts",
                .value_name = "/path/to/rule.fst",
                .identifier = 'b',
                .description = "It not empty, it contains a list of rule FST filenames."
                               "Multiple filenames are separated by a comma and they are "
                               "applied from left to right. An example value: "
                               "rule1.fst,rule2,fst,rule3.fst"},

        {.access_name = "max-num-sentences",
                .value_name = "2",
                .identifier = 'c',
                .description = "Maximum number of sentences that we process at a time. "
                               "This is to avoid OOM for very long input text. "
                               "If you set it to -1, then we process all sentences in a "
                               "single batch."},

        {.access_name = "vits-data-dir",
                .value_name = "/path/to/espeak-ng-data",
                .identifier = 'd',
                .description =
                "Path to espeak-ng-data. If it is given, --vits-lexicon is ignored"},

};



int test_sherpa_tts(){

    return 1;
}