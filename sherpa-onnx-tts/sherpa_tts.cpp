
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"



int test_sherpa_tts(){

    SherpaOnnxOfflineTtsConfig config;
    memset(&config, 0, sizeof(config));
    config.model.vits.model = "/Users/yang/CLionProjects/test_onnxruntime/sherpa-onnx-tts/vits-aishell3.int8.onnx";
    config.model.vits.lexicon = "/Users/yang/CLionProjects/test_onnxruntime/sherpa-onnx-tts/lexicon.txt";
    config.model.vits.tokens = "/Users/yang/CLionProjects/test_onnxruntime/sherpa-onnx-tts/tokens.txt";
    config.rule_fsts = "/Users/yang/CLionProjects/test_onnxruntime/sherpa-onnx-tts/rule.fst";
    config.model.vits.noise_scale = 0.667;
    config.model.vits.noise_scale_w = 0.8;
    config.model.vits.length_scale = 1.0;
    config.model.num_threads = 1;
    config.model.provider = "cpu";
    config.model.debug = 0;
    config.max_num_sentences = 2;

    int32_t sid = 0;
    const char *filename = strdup("/Users/yang/CLionProjects/test_onnxruntime/sherpa-onnx-tts/generated.wav");
    const char *text = "床前明月光，疑是地上霜。举头望明月，低头思故乡。";

    // 加载配置
    SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&config);
    // 生成音频
    const SherpaOnnxGeneratedAudio *audio =
            SherpaOnnxOfflineTtsGenerate(tts, text, sid, 1.0);
    // 保存文件
    SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate, filename);
    // 销毁模型
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    SherpaOnnxDestroyOfflineTts(tts);

//    fprintf(stderr, "Input text is: %s\n", text);
//    fprintf(stderr, "Speaker ID is is: %d\n", sid);
//    fprintf(stderr, "Saved to: %s\n", filename);

//    free((void *)filename);

    return 0;
//    return 1;
}