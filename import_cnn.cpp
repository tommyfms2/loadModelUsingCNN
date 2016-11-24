

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <sstream>


typedef std::vector<float> vf;

class Linear {
private:
    vf W;
    vf b;
    const int n_in, n_out;
    float relu(float x) {
        return (x > 0) ? x : 0;
    }
public:
    // Linearの方のコンストラクタ
    Linear(int in, int out) : n_in(in), n_out(out) {
        W.resize(n_in * n_out);
        b.resize(n_out);
    }
    void read(std::ifstream &ifs) {
        ifs.read((char*)W.data(), sizeof(float)*n_in * n_out);
        ifs.read((char*)b.data(), sizeof(float)*n_out);
    }
    vf get(vf x) {
        vf y(n_out);
        for (int i = 0; i < n_out; i++) {
            y[i] = 0.0;
            for (int j = 0; j < n_in; j++) {
                y[i] += W[i * n_in + j] * x[j];
            }
            y[i] += b[i];
        }
        return y;
    }
    vf get_relu(vf x){
        vf y = get(x);
        for (int i=0; i<n_out; i++) {
            y[i] = relu(y[i]);
        }
        return y;
    }
};

class Convolution2D {
private:
    vf C;
    vf b;
    const int n_in_channels, n_out_filter_num, n_filter_size, n_padding, n_stride;
    float relu(float x) {
        return (x > 0) ? x : 0;
    }
public:
    // Convの方のコンストラクタ
    Convolution2D(int infn, int outfn, int nfs, int np, int ns) : n_in_channels(infn), n_out_filter_num(outfn), n_filter_size(nfs), n_padding(np), n_stride(ns){
        C.resize(n_in_channels * n_out_filter_num * n_filter_size * n_filter_size);
        b.resize(n_out_filter_num);
    }
    void read(std::ifstream &ifs) {
        ifs.read((char*)C.data(), sizeof(float)*n_in_channels*n_out_filter_num*n_filter_size*n_filter_size);
        ifs.read((char*)b.data(), sizeof(float)*n_out_filter_num);
    }
    std::vector< vf > get(std::vector< vf > x) {
       
        int in_size = sqrt(x[0].size());
        int out_size = ((in_size+n_padding*2-n_filter_size)/n_stride+1);
        int half_of_filsize = n_filter_size >> 1;
        // paddingを考慮した入力を作る:　端は0
        std::vector< vf > tempX1( x.size() ,vf((in_size+n_padding*2)*(in_size+n_padding*2), 0));
        
        int c = 0;
        for (auto cx: x){
            for (int y=1; y<in_size+1; y++) {
                for (int x=1; x<in_size+1; x++) {
                    tempX1[c][y*(in_size+n_padding*2)+x] = cx[(y-1)*in_size+(x-1)];
                }
            }
            c++;
        }
        
        // 出力のベクトル
        std::vector< vf > y ( n_out_filter_num, vf( out_size*out_size ) );
        // フィルタごとのfor
        for (int fil_i=0; fil_i<n_out_filter_num; fil_i++) {
            // 出力画像のfor
            for (int in_y=n_padding; in_y<in_size+n_padding*2-1; in_y+=n_stride) {
                for (int in_x=n_padding; in_x<in_size+n_padding*2-1; in_x+=n_stride) {
                    y[fil_i][(in_y-n_padding)*out_size+(in_x-n_padding)] = 0;
                    // チャンネルのfor
                    for (int c=0; c<n_in_channels; c++) {
                        // フィルタ内のfor：畳込み
                        for (int k=0; k<n_filter_size; k++) {
                            for (int l=0; l<n_filter_size; l++) {
                                float pixVal = tempX1[c][(in_y+k-half_of_filsize)*(in_size+n_padding*2)+(in_x+l-half_of_filsize)];
                                float filVal = C[fil_i*(n_in_channels*n_filter_size*n_filter_size) +
                                                c*(n_filter_size*n_filter_size) +
                                                k*(n_filter_size) +
                                                 l];
                                y[fil_i][(in_y-n_padding)*out_size+(in_x-n_padding)] += pixVal * filVal;
                            }
                        }
                    }
                    // バイアスを足す
                    y[fil_i][(in_y-n_padding)*out_size+(in_x-n_padding)] += b[fil_i];
                }
            }
        }
        return y;
    }
    std::vector< vf > relu(std::vector< vf > x) {
        int j = 0;
        for (auto x1 : x){
            int i = 0;
            for (auto x2:x1){
                x[j][i] = relu(x2);
                i++;
            }
            j++;
        }
        return x;
    }
};


int argmax(vf &v) {
    float max = v[0];
    int max_i = 0;
    for (int i = 1; i < v.size(); i++) {
        if (max < v[i]) {
            max_i = i;
            max = v[i];
        }
    }
    return max_i;
}
std::vector<std::string> split(const std::string &str, char sep)
{
    std::vector<std::string> v;
    std::stringstream ss(str);
    std::string buffer;
    while( std::getline(ss, buffer, sep) ) {
        v.push_back(buffer);
    }
    return v;
}
std::vector< vf > load_test_data(int input_size){
    int input_num = (input_size+1) * (input_size+1);
    std::ifstream ifs("input_data.txt");
    std::string str;
    std::vector< vf > v (10, vf(25) );
    char index_to_finish10 = 0;
    if (ifs.fail())
    {
        std::cerr << "失敗" << std::endl;
        return v;
    }
    while (getline(ifs, str))
    {
        std::vector<std::string> nums = split(str, ',');
        int index_to_test = 0;
        for (auto vo: nums){
            v[index_to_finish10][index_to_test] = atoi( vo.c_str() );
            index_to_test++;
            if (index_to_test > 24) {
                break;
            }
        }
        index_to_finish10++;
        if (index_to_finish10 > 9) {
            break;
        }
    }
    
    return v;
}

int main(void) {
    const int n_size = 128;
    const int n_channels = 1;
    const int n_in = 576;
    const int n_units = 1024;
    const int n_out = 35;
    std::ifstream ifs("and.dat");
    
    std::cout << "set convolution" << std::endl;
    // conv1の準備 -> 読み込み -> 畳込み -> 出力層数 -> conv2の準備...
    Convolution2D conv1(n_channels, 64, 3, 1, 1); //in_channel,out_channel,filtersize,padding,stride
    Convolution2D conv2(64, 128, 3, 1, 1);
    Convolution2D conv3(128,256, 3, 1, 1);

    std::cout << "read ifs" << std::endl;
    conv1.read(ifs);
    conv2.read(ifs);
    conv3.read(ifs);

    std::cout << "set linear" << std::endl;
    Linear fc4(6400, 4096), fc5(4096, 1024), fc6(1024, 35);
    fc4.read(ifs);
    fc5.read(ifs);
    fc6.read(ifs);
    
    std::vector< vf > test_datas = load_test_data(4);
    

    std::cout << "prediction" << std::endl;
    // 全部のデータ
    int tesi = 0;
    for (auto test_data : test_datas){
        // このループでpredictするためのデータを作る. 今回はchannelが1なので(1,25)の2次元
        std::vector< vf > x2(1, vf(25) );
        int i = 0;
        for (auto v2:test_data){
            x2[0][i] = v2/1023;
            i++;
        }
        
        std::vector< vf > h = conv3.relu(conv3.get(conv2.relu(conv2.get(conv1.relu(conv1.get(x2))))));
        vf x3;
        for(auto h1:h){
            for(auto h2:h1){
                x3.push_back(h2);
            }
        }
        vf y = fc6.get(fc5.get_relu(fc4.get_relu(x3)));
        std::cout << tesi <<" ";
        std::cout << "["<<x2[0][0]<<" "<<x2[0][1]<<" "<<x2[0][2]<<", ... ,"<<x2[0][22]<<" "<<x2[0][23]<<" "<<x2[0][24]<<"]"<<std::endl;
        std::cout << "  ---" << argmax(y) << "---" << std::endl;
        for (auto yy:y){
            std::cout << yy << " ";
        }
        std::cout << std::endl;
        tesi++;
    }
}


