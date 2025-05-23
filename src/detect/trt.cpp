// Copyright 2025 Zikang Xie
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "detect/trt.hpp"

#include <fstream>

#include "NvOnnxParser.h"
#include "common/logger.hpp"
#include "cuda_runtime_api.h"
#include "common/gobal.hpp"


// #include <logger.h>
#define TRT_ASSERT(expr)                                                \
  do {                                                                  \
    if (!(expr)) {                                                      \
      fmt::print(fmt::fg(fmt::color::red), "assert fail: '" #expr "'"); \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)


static const int INPUT_W = 416;        // Width of input
static const int INPUT_H = 416;        // Height of input
static constexpr int NUM_CLASSES = 8;  // Number of classes
static constexpr int NUM_COLORS = 4;   // Number of color
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;
// 辅助函数：生成网格和步长

static void generate_grids_and_stride(
  std::vector<int> & strides, std::vector<GridAndStride> & grid_strides)
{
  for (auto stride : strides) {
    int num_grid_w = 416 / stride;
    int num_grid_h = 416 / stride;
    for (int g1 = 0; g1 < num_grid_h; g1++) {
      for (int g0 = 0; g0 < num_grid_w; g0++) {
        grid_strides.push_back(GridAndStride{g0, g1, stride});
      }
    }
  }
}
static cv::Mat letterbox(
  const cv::Mat & img, Eigen::Matrix3f & transform_matrix,
  std::vector<int> new_shape = {INPUT_W, INPUT_H})
{
  // Get current image shape [height, width]
  int img_h = img.rows;
  int img_w = img.cols;

  // Compute scale ratio(new / old) and target resized shape
  float scale = std::min(new_shape[1] * 1.0 / img_h, new_shape[0] * 1.0 / img_w);
  int resize_h = static_cast<int>(round(img_h * scale));
  int resize_w = static_cast<int>(round(img_w * scale));

  // Compute padding
  int pad_h = new_shape[1] - resize_h;
  int pad_w = new_shape[0] - resize_w;

  // Resize and pad image while meeting stride-multiple constraints
  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(resize_w, resize_h));

  // divide padding into 2 sides
  float half_h = pad_h * 1.0 / 2;
  float half_w = pad_w * 1.0 / 2;

  // Compute padding boarder
  int top = static_cast<int>(round(half_h - 0.1));
  int bottom = static_cast<int>(round(half_h + 0.1));
  int left = static_cast<int>(round(half_w - 0.1));
  int right = static_cast<int>(round(half_w + 0.1));

  /* clang-format off */
  /* *INDENT-OFF* */

  // Compute point transform_matrix
  transform_matrix << 1.0 / scale, 0, -half_w / scale,
                      0, 1.0 / scale, -half_h / scale,
                      0, 0, 1;

  /* *INDENT-ON* */
  /* clang-format on */

  // Add border
  cv::copyMakeBorder(
    resized_img, resized_img, top, bottom, left, right, cv::BORDER_CONSTANT,
    cv::Scalar(114, 114, 114));

  return resized_img;
}
/**
 * @brief Calculate intersection area between two objects.
 * @param a Object a.
 * @param b Object b.
 * @return Area of intersection.
 */
static inline float intersection_area(const ArmorObject & a, const ArmorObject & b)
{
  cv::Rect_<float> inter = a.box & b.box;
  return inter.area();
}

static void nms_merge_sorted_bboxes(
  std::vector<ArmorObject> & faceobjects, std::vector<int> & indices, float nms_threshold)
{
  indices.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].box.area();
  }

  for (int i = 0; i < n; i++) {
    ArmorObject & a = faceobjects[i];

    int keep = 1;
    for (int indice : indices) {
      ArmorObject & b = faceobjects[indice];

      // intersection over union
      float inter_area = intersection_area(a, b);
      float union_area = areas[i] + areas[indice] - inter_area;
      float iou = inter_area / union_area;
      if (iou > nms_threshold || isnan(iou)) {
        keep = 0;
        // Stored for Merge
        if (
          a.number == b.number && a.color == b.color && iou > MERGE_MIN_IOU &&
          abs(a.prob - b.prob) < MERGE_CONF_ERROR) {
          for (int i = 0; i < 4; i++) {
            b.pts.push_back(a.pts[i]);
          }
        }
        // cout<<b.pts_x.size()<<endl;
      }
    }

    if (keep) {
      indices.push_back(i);
    }
  }
}
bool AdaptedTRTModule::extractImage(const cv::Mat & src, ArmorObject & armor) {
  // Constants
  static const int light_length = 12;
  static const int warp_height = 28;
  static const int small_armor_width = 32;
  static const int large_armor_width = 54;
  static const cv::Size roi_size(20, 28);
  static const cv::Size input_size(28, 28);

  if (src.empty() || src.cols < 10 || src.rows < 10) {
     // std::cerr << "Source image is empty or too small" << std::endl;
      return false;
  }

  std::vector<cv::Point2f> pts_vec(std::begin(armor.pts), std::end(armor.pts));
  cv::Rect bbox = cv::boundingRect(pts_vec);

  float expand_ratio_w = 2.0f;
  float expand_ratio_h = 1.5f;
  int new_width = static_cast<int>(bbox.width * expand_ratio_w);
  int new_height = static_cast<int>(bbox.height * expand_ratio_h);
  int new_x = static_cast<int>(bbox.x - (new_width - bbox.width) / 2);
  int new_y = static_cast<int>(bbox.y - (new_height - bbox.height) / 2);

  // 边界检查
  new_x = std::max(0, new_x);
  new_y = std::max(0, new_y);
  if (new_x + new_width > src.cols) new_width = src.cols - new_x;
  if (new_y + new_height > src.rows) new_height = src.rows - new_y;

  if (new_width <= 0 || new_height <= 0) {
     // std::cerr << "Expanded ROI is invalid" << std::endl;
      return false;
  }

  cv::Rect expanded_rect(new_x, new_y, new_width, new_height);
  armor.new_x = new_x;
  armor.new_y = new_y;

  // 截取 ROI 区域
  cv::Mat litroi_color = src(expanded_rect).clone();
  if (litroi_color.empty()) {
      //std::cerr << "ROI color image is empty" << std::endl;
      return false;
  }

  cv::Mat litroi_gray;
  cv::cvtColor(litroi_color, litroi_gray, cv::COLOR_RGB2GRAY);

  armor.whole_rgb_img = litroi_color;
  armor.whole_gray_img = litroi_gray;

  cv::Mat litroi_bin;
  cv::threshold(litroi_gray, litroi_bin, binary_thres_, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  armor.whole_binary_img = litroi_bin;

  // Perspective warp
  cv::Point2f lights_vertices[4] = {armor.pts[0], armor.pts[1], armor.pts[2], armor.pts[3]};
  int top_light_y = (warp_height - light_length) / 2 - 1;
  int bottom_light_y = top_light_y + light_length;
  int warp_width = (armor.number == ArmorNumber::NO1 || armor.number == ArmorNumber::BASE) ? small_armor_width : large_armor_width;

  cv::Point2f target_vertices[4] = {
      cv::Point(0, bottom_light_y),
      cv::Point(0, top_light_y),
      cv::Point(warp_width - 1, top_light_y),
      cv::Point(warp_width - 1, bottom_light_y),
  };

  cv::Mat warp_mat = cv::getPerspectiveTransform(lights_vertices, target_vertices);
  cv::Mat number_image;
  cv::warpPerspective(src, number_image, warp_mat, cv::Size(warp_width, warp_height));

  if (number_image.empty() || number_image.cols < roi_size.width || number_image.rows < roi_size.height) {
     // std::cerr << "Warped number image is invalid" << std::endl;
      return false;
  }

  // 截取 ROI 并处理
  cv::Rect number_roi((warp_width - roi_size.width) / 2, 0, roi_size.width, roi_size.height);
  if ((number_roi.x + number_roi.width > number_image.cols) || 
      (number_roi.y + number_roi.height > number_image.rows)) {
     // std::cerr << "ROI for number image is out of bounds" << std::endl;
      return false;
  }

  number_image = number_image(number_roi).clone();
  cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY);
  cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  cv::resize(number_image, number_image, input_size);
  cv::Mat flipped_image;
  cv::flip(number_image, flipped_image, 0);
  armor.number_img = flipped_image;

  return true;
}


  bool AdaptedTRTModule::isLight(const Light &light) noexcept {
    // The ratio of light (short side / long side)
    float ratio = light.width / light.length;
    bool ratio_ok = light_params_.min_ratio < ratio && ratio < light_params_.max_ratio;
  
    bool angle_ok = light.tilt_angle < light_params_.max_angle;
  
    bool is_light = ratio_ok && angle_ok;
  
  
    return is_light;
  }
  std::vector<Light> AdaptedTRTModule::findLights(const cv::Mat &rgb_img,
    const cv::Mat &binary_img, ArmorObject &armor) noexcept
  {
      using std::vector;
      vector<vector<cv::Point>> contours;
      vector<cv::Vec4i> hierarchy;
  
  
      cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  
      vector<Light> all_lights;
  
      for (const auto &contour : contours) {
          if (contour.size() < 6) continue;
  
          auto light = Light(contour);
          if (isLight(light)) {
              all_lights.emplace_back(light);
          }
      }
  
      std::sort(all_lights.begin(), all_lights.end(), [](const Light &l1, const Light &l2) {
          return l1.center.x < l2.center.x;
      });
  
      // 更新 armor 内的信息
      armor.lights = all_lights;
      if (armor.lights.empty()) return all_lights;
    if (armor.whole_gray_img.empty()) {return all_lights;
    
    }
      double zero_x = armor.new_x;
      double zero_y = armor.new_y;
      for (auto& light : armor.lights) {
        
        light.top.x += zero_x;
        light.top.y += zero_y;
        light.center.x += zero_x;
        light.center.y += zero_y;

        light.bottom.x += zero_x;
        light.bottom.y += zero_y;
        light.axis.x += zero_x;
        light.axis.y += zero_y;


    }
      std::vector<std::pair<const Light*, double>> light_distances;
      cv::Point2f armor_center = (armor.pts[0] + armor.pts[1] + armor.pts[2] + armor.pts[3]) * 0.25;
      for (const auto& light : armor.lights) {
          double dist = cv::norm(light.center - armor_center);
          light_distances.emplace_back(&light, dist);
      }

      // Step 3: 按距离排序，选择最近两个灯条
      std::sort(light_distances.begin(), light_distances.end(), [](const auto& a, const auto& b) {
          return a.second < b.second;
      });
      if (light_distances.size() >= 2) {
          const Light* l1 = light_distances[0].first;
          const Light* l2 = light_distances[1].first;
      
          
      }


      // Step 4: 构建 candidates，只保留两个灯条的 top/bottom
      std::vector<cv::Point2f> candidates;
      for (int i = 0; i < std::min(2, (int)light_distances.size()); ++i) {
          const auto* light = light_distances[i].first;
          candidates.push_back(light->top);
          candidates.push_back(light->bottom);
          
      }


      double w = cv::norm(armor.pts[0] - armor.pts[1]);
      double h = cv::norm(armor.pts[0] - armor.pts[3]);
      double size_scale = w + h;

      std::vector<cv::Point2f> selected_pts(4, cv::Point2f(-1, -1));
      std::vector<int> selected_indices(4, -1); 

  
      for (int i = 0; i < 4; ++i) {
          double min_dist = DBL_MAX;
          int best_match = -1;

          double test_result = cv::pointPolygonTest(armor.pts, armor.pts[i], false);
          double dist_threshold = (test_result >= 0) ? (0.15 * size_scale) : (0.25 * size_scale);

          for (size_t j = 0; j < candidates.size(); ++j) {
              double dist = cv::norm(armor.pts[i] - candidates[j]);
              if (dist < min_dist) {
                  min_dist = dist;
                  best_match = static_cast<int>(j);
              }
          }

          if (best_match != -1 && min_dist < dist_threshold) {
              selected_pts[i] = candidates[best_match];
              selected_indices[i] = best_match;
          }
      }


      for (const auto& pt : selected_pts) {
          if (pt.x >= 0 && pt.y >= 0) {
              auto it = std::find(candidates.begin(), candidates.end(), pt);
              if (it != candidates.end()) candidates.erase(it);
          }
          armor.pts_binary.push_back(pt);
      }


      armor.is_ok = true;
      for (const auto& pt : armor.pts_binary) {
          if (pt.x < 0 || pt.y < 0) {
              armor.is_ok = false;
              break;
          }
      }


      if (std::count_if(armor.pts_binary.begin(), armor.pts_binary.end(), [](const cv::Point2f& p) {
          return p.x >= 0 && p.y >= 0;
      }) != 4) {
          armor.is_ok = false;
      }


      if (!armor.is_ok) {
          armor.pts_binary.clear();
          
      }


      //armor.is_ok = !armor.lights.empty();
  
      return all_lights;
  }
    void AdaptedTRTModule::detect(ArmorObject & armor)
  { 
    findLights(armor.whole_rgb_img,armor.whole_binary_img,armor);
   

    
  }
// 构造函数：初始化参数并构建引擎
AdaptedTRTModule::AdaptedTRTModule(const std::string & onnx_path, const Params & params, double expand_ratio_w, double expand_ratio_h, int binary_thres, LightParams light_params,std::string classify_model_path,std::string classify_label_path)
: params_(params), engine_(nullptr), context_(nullptr), output_buffer_(nullptr), runtime_(nullptr),expand_ratio_h_(expand_ratio_h),expand_ratio_w_(expand_ratio_w),binary_thres_(binary_thres),light_params_(light_params),classify_label_path_(classify_label_path),classify_model_path_(classify_model_path)
{
  buildEngine(onnx_path);
  TRT_ASSERT(context_ = engine_->createExecutionContext());
  TRT_ASSERT((input_idx_ = engine_->getBindingIndex("images")) == 0);
  TRT_ASSERT((output_idx_ = engine_->getBindingIndex("output")) == 1);

  auto input_dims = engine_->getBindingDimensions(input_idx_);
  auto output_dims = engine_->getBindingDimensions(output_idx_);
  input_sz_ = input_dims.d[1] * input_dims.d[2] * input_dims.d[3];
  output_sz_ = output_dims.d[1] * output_dims.d[2];
  TRT_ASSERT(cudaMalloc(&device_buffers_[input_idx_], input_sz_ * sizeof(float)) == 0);
  TRT_ASSERT(cudaMalloc(&device_buffers_[output_idx_], output_sz_ * sizeof(float)) == 0);
  output_buffer_ = new float[output_sz_];
  TRT_ASSERT(cudaStreamCreate(&stream_) == 0);
  thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 100);
}

AdaptedTRTModule::~AdaptedTRTModule()
{
  delete[] output_buffer_;
  cudaStreamDestroy(stream_);
  cudaFree(device_buffers_[output_idx_]);
  cudaFree(device_buffers_[input_idx_]);
  if (context_) context_->destroy();
  if (engine_) engine_->destroy();
  if (runtime_) runtime_->destroy();  
  thread_pool_.reset();
    if (thread_pool_) {
        thread_pool_->waitUntilEmpty(); 
    }
}

void AdaptedTRTModule::buildEngine(const std::string & onnx_path)
{
  std::string engine_path = onnx_path.substr(0, onnx_path.find_last_of('.')) + ".engine";
  std::ifstream engine_file(engine_path, std::ios::binary);
  if (engine_file.good()) {
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);
    engine_file.close();

    runtime_ = nvinfer1::createInferRuntime(g_logger_);  // ✅ 作为成员变量保存
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (engine_ != nullptr) {
      WUST_INFO("TRT") << "Load engine from " << engine_path << " successfully.";
      return;
    }
  }

  // 构建新引擎
  auto builder = nvinfer1::createInferBuilder(g_logger_);
  const auto explicit_batch =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = builder->createNetworkV2(explicit_batch);
  auto parser = nvonnxparser::createParser(*network, g_logger_);
  parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));

  auto config = builder->createBuilderConfig();
  if (builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  engine_ = builder->buildEngineWithConfig(*network, *config);

  // 保存引擎
  auto serialized_engine = engine_->serialize();
  std::ofstream out_file(engine_path, std::ios::binary);
  out_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
  out_file.close();
  serialized_engine->destroy();

  // 反序列化仍然需要 runtime_
  if (!runtime_) {
    runtime_ = nvinfer1::createInferRuntime(g_logger_);
  }

  // 清理
  parser->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();

  WUST_INFO("TRT") << "Build engine from " << onnx_path << " successfully.";
}

void AdaptedTRTModule::setCallback(DetectorCallback callback) { infer_callback_ = callback; }


// 推理函数
bool AdaptedTRTModule::processCallback(
    const cv::Mat resized_img, Eigen::Matrix3f transform_matrix, int64_t timestamp_nanosec,
    const cv::Mat & src_img)
{
  // 预处理：Letterbox 缩放
  // cv::Mat resized;
  // cv::resize(image, resized, cv::Size(params_.input_w, params_.input_h));

//   Eigen::Matrix3f transform_matrix;  // transform matrix from resized image to source image.
//   cv::Mat resized = letterbox(image, transform_matrix);

  // cv::Mat blob =
  //   cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(), cv::Scalar(0, 0, 0), true);
//   auto end = std::chrono::steady_clock::now();
//     std::chrono::steady_clock::time_point timestamp_timepoint = 
//     std::chrono::steady_clock::time_point(std::chrono::nanoseconds(timestamp_nanosec));
// auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - timestamp_timepoint);

// WUST_INFO("TRT") << "Detect time: " << duration.count() << " ms";

  cv::Mat blob =
    cv::dnn::blobFromImage(resized_img, 1., cv::Size(INPUT_W, INPUT_H), cv::Scalar(0, 0, 0), true);
  // 拷贝数据到显存
  cudaMemcpyAsync(
    device_buffers_[input_idx_], blob.ptr<float>(), input_sz_ * sizeof(float),
    cudaMemcpyHostToDevice, stream_);
  context_->enqueueV2(device_buffers_, stream_, nullptr);
  cudaMemcpyAsync(
    output_buffer_, device_buffers_[output_idx_], output_sz_ * sizeof(float),
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  std::vector<ArmorObject> objs_tmp, objs_result;
  std::vector<cv::Rect> rects;
  std::vector<float> scores;
  // 后处理
  objs_result=postprocess(
    objs_tmp, scores, rects, output_buffer_, output_sz_ / 21, transform_matrix); 
  


  for (auto & armor : objs_result) {
      if(armor.color == ArmorColor::NONE||armor.color == ArmorColor::PURPLE)
      {
        continue;
      }
      if (detect_color_ == 0 && armor.color != ArmorColor::RED) {
        continue;
      } else if (detect_color_ == 1 && armor.color != ArmorColor::BLUE) {
        continue;
      }
      
     if( extractImage(src_img, armor))
     {
      classifyNumber(armor);
      detect(armor);
     }
      
     
     
      
    
    }  
  

    

    if (this->infer_callback_) {
        this->infer_callback_(objs_result, timestamp_nanosec, src_img);
        return true;
      }
      

  return true; 
}
void AdaptedTRTModule::initNumberClassifier()
{
  // 加载数字识别模型
  const std::string model_path = classify_model_path_;
  number_net_ = cv::dnn::readNetFromONNX(model_path);

  // 检查模型是否成功加载
  if (number_net_.empty()) {
    std::cerr << "Failed to load number classifier model from " << model_path << std::endl;
    std::exit(EXIT_FAILURE);  // 模型加载失败，退出程序
  } else {
    std::cout << "Successfully loaded number classifier model from " << model_path << std::endl;
  }

  // 加载标签
  const std::string label_path = classify_label_path_;
  std::ifstream label_file(label_path);
  std::string line;

  // 清空之前的标签
  class_names_.clear();

  // 读取标签文件
  while (std::getline(label_file, line)) {
    class_names_.push_back(line);
  }

  // 检查标签是否成功加载
  if (class_names_.empty()) {
    std::cerr << "Failed to load labels from " << label_path << std::endl;
    std::exit(EXIT_FAILURE);  // 标签加载失败，退出程序
  } else {
    std::cout << "Successfully loaded " << class_names_.size() << " labels from " << label_path
              << std::endl;
  }
}
bool AdaptedTRTModule::classifyNumber(ArmorObject & armor)  {
  // Normalize

  static thread_local std::unique_ptr<cv::dnn::Net> thread_net;
    if (!thread_net) {
    thread_net = std::make_unique<cv::dnn::Net>(cv::dnn::readNetFromONNX(classify_model_path_));
    if (thread_net->empty()) {
      std::cerr << "Failed to load thread-local number classifier model." << std::endl;
      return false;
    }
  }
  cv::Mat image = armor.number_img.clone();
  image=image/255.0;

  // Create blob from image
  cv::Mat blob;
  cv::dnn::blobFromImage(image, blob);
  
  // Set the input blob for the neural network
 // mutex_.lock();
  thread_net->setInput(blob);

  // Forward pass the image blob through the model
  cv::Mat outputs = thread_net->forward().clone();
 // mutex_.unlock();

  // Decode the output
  double confidence;
  cv::Point class_id_point;
  minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
  int label_id = class_id_point.x;

  armor.confidence = confidence;
    static const std::map<int, ArmorNumber> label_to_armor_number = {
    {0, ArmorNumber::NO1}, {1, ArmorNumber::NO2}, {2, ArmorNumber::NO3},
    {3, ArmorNumber::NO4}, {4, ArmorNumber::NO5}, {5, ArmorNumber::OUTPOST},
    {6, ArmorNumber::SENTRY}, {7, ArmorNumber::BASE}
  };
    if (label_id < 8 && label_to_armor_number.find(label_id) != label_to_armor_number.end()) {
    armor.number = label_to_armor_number.at(label_id);
    return true;
  } else {
    armor.confidence = 0;
    return false;
  }
  
}


// 后处理函数
std::vector<ArmorObject> AdaptedTRTModule::postprocess(
  std::vector<ArmorObject> & output_objs, std::vector<float> & scores,
  std::vector<cv::Rect> & rects, const float * output, int num_detections,
  const Eigen::Matrix<float, 3, 3> & transform_matrix)
{
  std::vector<int> strides = {8, 16, 32};
  std::vector<GridAndStride> grid_strides;
  generate_grids_and_stride(strides, grid_strides);
 
  for (int i = 0; i < num_detections; ++i) {
    const float * det = output + i * 21;
    float conf = det[8];
   
    if (conf < params_.conf_threshold) continue;

    // 解析坐标
    int grid0 = grid_strides[i].grid0;
    int grid1 = grid_strides[i].grid1;
    int stride = grid_strides[i].stride;
    // // 第一个点（左上）
    // box.pts[0].x = (det[0] + grid0) * stride;
    // box.pts[0].y = (det[1] + grid1) * stride;

    // // 第二个点（右上）
    // box.pts[1].x = (det[2] + grid0) * stride;
    // box.pts[1].y = (det[3] + grid1) * stride;

    // // 第三个点（右下）
    // box.pts[2].x = (det[4] + grid0) * stride;
    // box.pts[2].y = (det[5] + grid1) * stride;

    // // 第四个点（左下）
    // box.pts[3].x = (det[6] + grid0) * stride;
    // box.pts[3].y = (det[7] + grid1) * stride;
    cv::Point color_id, num_id;

    float x_1 = (det[0] + grid0) * stride;
    float y_1 = (det[1] + grid1) * stride;
    float x_2 = (det[2] + grid0) * stride;
    float y_2 = (det[3] + grid1) * stride;
    float x_3 = (det[4] + grid0) * stride;
    float y_3 = (det[5] + grid1) * stride;
    float x_4 = (det[6] + grid0) * stride;
    float y_4 = (det[7] + grid1) * stride;

    Eigen::Matrix<float, 3, 4> apex_norm;
    Eigen::Matrix<float, 3, 4> apex_dst;

    apex_norm << x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, 1, 1, 1, 1;

    apex_dst = transform_matrix * apex_norm;

    ArmorObject obj;

    obj.pts.resize(4);

    obj.pts[0] = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
    obj.pts[1] = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
    obj.pts[2] = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
    obj.pts[3] = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));

    auto rect = cv::boundingRect(obj.pts);

    obj.box = rect;
    // obj.color = static_cast<ArmorColor>(color_id.x);
    // obj.number = static_cast<ArmorNumber>(num_id.x);
    obj.prob = conf;

    // 解析颜色和类别
    obj.color =
      static_cast<ArmorColor>(std::max_element(det + 9, det + 9 + NUM_COLORS) - (det + 9));
    obj.number = static_cast<ArmorNumber>(
      std::max_element(det + 9 + NUM_COLORS, det + 9 + NUM_COLORS + NUM_CLASSES) -
      (det + 9 + NUM_COLORS));
    // box.confidence = conf;

    rects.push_back(rect);
    scores.push_back(conf);
    output_objs.push_back(std::move(obj));
  }

  // TopK
  std::sort(
    output_objs.begin(), output_objs.end(),
    [](const ArmorObject & a, const ArmorObject & b) { return a.prob > b.prob; });
  if (output_objs.size() > static_cast<size_t>(params_.top_k)) {
    output_objs.resize(params_.top_k);
  }
  std::vector<int> indices;
  std::vector<ArmorObject> objs_result;
  // cv::dnn::NMSBoxes(rects, scores, params_.conf_threshold, params_.nms_threshold, indices);
  nms_merge_sorted_bboxes(output_objs, indices, params_.nms_threshold);

  for (size_t i = 0; i < indices.size(); i++) {
    objs_result.push_back(std::move(output_objs[indices[i]]));

    if (objs_result[i].pts.size() >= 8) {
      auto n = objs_result[i].pts.size();
      cv::Point2f pts_final[4];

      for (size_t j = 0; j < n; j++) {
        pts_final[j % 4] += objs_result[i].pts[j];
      }

      objs_result[i].pts.resize(4);
      for (int j = 0; j < 4; j++) {
        pts_final[j].x /= static_cast<float>(n) / 4.0;
        pts_final[j].y /= static_cast<float>(n) / 4.0;
        objs_result[i].pts[j] = pts_final[j];
      }
    }
  }


  return objs_result;
}

void AdaptedTRTModule::pushInput(const cv::Mat& rgb_img, int64_t timestamp_nanosec) {
    if (rgb_img.empty()) {
      return;
    }
  
  
  
    Eigen::Matrix3f transform_matrix;
    cv::Mat resized_img = letterbox(rgb_img, transform_matrix);
  
  
  
    thread_pool_->enqueue([this, resized_img, transform_matrix, timestamp_nanosec, rgb_img]() {
      this->processCallback(resized_img, transform_matrix, timestamp_nanosec, rgb_img);
      
    });
  }
