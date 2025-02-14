
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <fstream>

#include "sherpa-onnx/c-api/c-api.h"


bool isValidUtf8(const std::string& str) {
    int c,i,ix,n,j;
    for (i=0, ix=str.length(); i < ix; i++) {
        c = (unsigned char)str[i];
        if (c<=0x7f) { // 0xxxxxxx
            continue;
        } else if ((c & 0xE0) == 0xC0) { // 110xxxxx
            n=1;
        } else if ((c & 0xF0) == 0xE0) { // 1110xxxx
            n=2;
        } else if ((c & 0xF8) == 0xF0) { // 11110xxx
            n=3;
        } else {
            return false;
        }
        for (j=0; j<n && i<ix; j++) { // n bytes matching 10xxxxxx follow ?
            if ((++i == ix) || (((unsigned char)str[i] & 0xC0) != 0x80))
                return false;
        }
    }
    return true;
}


int test_sherpa_tts(){

    SherpaOnnxOfflineTtsConfig config;
    memset(&config, 0, sizeof(config));
    config.model.vits.model = "/Users/yang/CLionProjects/test_onnxruntime/sherpa-onnx-tts/vits-aishell3.onnx";
    config.model.vits.lexicon = "/Users/yang/CLionProjects/test_onnxruntime/sherpa-onnx-tts/lexicon.txt";
    config.model.vits.tokens = "/Users/yang/CLionProjects/test_onnxruntime/sherpa-onnx-tts/tokens.txt";
    config.rule_fsts = "/Users/yang/CLionProjects/test_onnxruntime/sherpa-onnx-tts/rule.fst";
    config.model.vits.noise_scale = 0.667;
    config.model.vits.noise_scale_w = 0.8;
    config.model.vits.length_scale = 1.0;
    config.model.num_threads = 1;
    config.model.provider = "cpu";
    config.model.debug = 1;
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

    std::ifstream file(config.model.vits.lexicon);
    std::string line;
    std::unordered_map<std::string, std::string> dict;

    if (file.is_open()) {
        while (getline(file, line)) {
            std::istringstream iss(line);
            std::string key, value, temp;

            // 读取第一列作为key
            iss >> key;

            // 读取剩下的部分作为value
            while (iss >> temp) {
                value += temp + " ";
            }
            // 删除最后一个多余的空格
            if (!value.empty()) value.pop_back();

            // 插入到字典中
            dict[key] = value;
            if (isValidUtf8(key)) {
                std::cout << key << ": the string is valid UTF-8." << std::endl;
            } else {
                std::cout << key << ": the string is not valid UTF-8." << std::endl;
            }
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
        return 1; // 文件打开失败
    }
    // 打印字典，验证结果
//    for (const auto &pair : dict) {
//        std::cout << pair.first << ": " << pair.second << std::endl;
//    }

    std::string output;
    output += dict["床"];
    output += dict["前"];
    output += dict["明"];
    output += dict["月"];
    output += dict["光"];
    std::cout << output << std::endl;
    const char* output_c = output.c_str();

    return 1;
}