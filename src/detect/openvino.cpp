#include "detect/openvino.hpp"
#include "common/logger.hpp"
#include <opencv2/highgui.hpp>
#include <functional>
static const int INPUT_W = 416;        // Width of input
static const int INPUT_H = 416;        // Height of input
static constexpr int NUM_CLASSES = 8;  // Number of classes
static constexpr int NUM_COLORS = 4;   // Number of color
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;
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
  
static void generate_grids_and_stride(
    const int target_w, const int target_h, std::vector<int> & strides,
    std::vector<GridAndStride> & grid_strides)
  {
    for (auto stride : strides) {
      int num_grid_w = target_w / stride;
      int num_grid_h = target_h / stride;
  
      for (int g1 = 0; g1 < num_grid_h; g1++) {
        for (int g0 = 0; g0 < num_grid_w; g0++) {
          grid_strides.emplace_back(GridAndStride{g0, g1, stride});
        }
      }
    }
  }
  static void generate_proposals(
    std::vector<ArmorObject> & output_objs, std::vector<float> & scores,
    std::vector<cv::Rect> & rects, const cv::Mat & output_buffer,
    const Eigen::Matrix<float, 3, 3> & transform_matrix, float conf_threshold,
    std::vector<GridAndStride> grid_strides)
  {
    const int num_anchors = grid_strides.size();
  
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
      float confidence = output_buffer.at<float>(anchor_idx, 8);
      if (confidence < conf_threshold) {
        continue;
      }
  
      const int grid0 = grid_strides[anchor_idx].grid0;
      const int grid1 = grid_strides[anchor_idx].grid1;
      const int stride = grid_strides[anchor_idx].stride;
  
      double color_score, num_score;
      cv::Point color_id, num_id;
      cv::Mat color_scores = output_buffer.row(anchor_idx).colRange(9, 9 + NUM_COLORS);
      cv::Mat num_scores =
        output_buffer.row(anchor_idx).colRange(9 + NUM_COLORS, 9 + NUM_COLORS + NUM_CLASSES);
      // Argmax
      cv::minMaxLoc(color_scores, NULL, &color_score, NULL, &color_id);
      cv::minMaxLoc(num_scores, NULL, &num_score, NULL, &num_id);
  
      float x_1 = (output_buffer.at<float>(anchor_idx, 0) + grid0) * stride;
      float y_1 = (output_buffer.at<float>(anchor_idx, 1) + grid1) * stride;
      float x_2 = (output_buffer.at<float>(anchor_idx, 2) + grid0) * stride;
      float y_2 = (output_buffer.at<float>(anchor_idx, 3) + grid1) * stride;
      float x_3 = (output_buffer.at<float>(anchor_idx, 4) + grid0) * stride;
      float y_3 = (output_buffer.at<float>(anchor_idx, 5) + grid1) * stride;
      float x_4 = (output_buffer.at<float>(anchor_idx, 6) + grid0) * stride;
      float y_4 = (output_buffer.at<float>(anchor_idx, 7) + grid1) * stride;
  
      Eigen::Matrix<float, 3, 4> apex_norm;
      Eigen::Matrix<float, 3, 4> apex_dst;
  
      /* clang-format off */
      /* *INDENT-OFF* */
      apex_norm << x_1, x_2, x_3, x_4,
                  y_1, y_2, y_3, y_4,
                  1,   1,   1,   1;
      /* *INDENT-ON* */
      /* clang-format on */
  
      apex_dst = transform_matrix * apex_norm;
  
      ArmorObject obj;
  
      obj.pts.resize(4);
  
      obj.pts[0] = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
      obj.pts[1] = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
      obj.pts[2] = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
      obj.pts[3] = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
  
      auto rect = cv::boundingRect(obj.pts);
  
      obj.box = rect;
      obj.color = static_cast<ArmorColor>(color_id.x);
      obj.number = static_cast<ArmorNumber>(num_id.x);
      obj.prob = confidence;
  
      rects.push_back(rect);
      scores.push_back(confidence);
      output_objs.push_back(std::move(obj));
    }
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
      for (size_t j = 0; j < indices.size(); j++) {
        ArmorObject & b = faceobjects[indices[j]];
  
        // intersection over union
        float inter_area = intersection_area(a, b);
        float union_area = areas[i] + areas[indices[j]] - inter_area;
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
OpenVino::OpenVino(
    const std::filesystem::path & model_path, const std::string & classify_model_path,const
    std::string & classify_label_path, const std::string & device_name,const LightParams &l, float conf_threshold,
    int top_k, float nms_threshold , float expand_ratio_w , float expand_ratio_h , int binary_thres_ , bool auto_init )
  : 
    light_params_(l),
    model_path_(model_path),
    classify_model_path_(classify_model_path),
    classify_label_path_(classify_label_path),
    device_name_(device_name),
    conf_threshold_(conf_threshold),
    top_k_(top_k),
    nms_threshold_(nms_threshold),
    number_threshold_(0.2),
    binary_thres_(binary_thres_),
    expand_ratio_w_(expand_ratio_w),
    expand_ratio_h_(expand_ratio_h)
  {
    // 初始化数字识别模型和标签
    initNumberClassifier();
    if (auto_init) {
      init();
    }
    
  }
  void OpenVino::initNumberClassifier()
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
  
  void OpenVino::init()
  {
    if (ov_core_ == nullptr) {
      ov_core_ = std::make_unique<ov::Core>();
    }
  
    auto model = ov_core_->read_model(model_path_);
  
    // Set infer type
    ov::preprocess::PrePostProcessor ppp(model);
    // Set input output precision
    ppp.input().tensor().set_element_type(ov::element::f32);
    ppp.output().tensor().set_element_type(ov::element::f32);
  
    // Compile model
    compiled_model_ = std::make_unique<ov::CompiledModel>(ov_core_->compile_model(
      model, device_name_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));
  
    strides_ = {8, 16, 32};
    generate_grids_and_stride(INPUT_W, INPUT_H, strides_, grid_strides_);
    thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency(), 100);
    
  
  }
  OpenVino::~OpenVino()
  { thread_pool_.reset();
    if (thread_pool_) {
      thread_pool_->waitUntilEmpty();
    }
  }
//   void OpenVino::extractNumberImage(const cv::Mat & src, ArmorObject & armor)
// {
//   // 光条长度和装甲板尺寸参数
//   const int light_length = 12;
//   const int warp_height = 28;
//   const int small_armor_width = 32;
//   const int large_armor_width = 54;
//   const cv::Size roi_size(20, 28);

//   // 判断装甲板类型
//   bool is_large = (armor.number == ArmorNumber::NO1 || armor.number == ArmorNumber::BASE);

//   // 计算外接矩形并扩展
//   std::vector<cv::Point2f> pts_vec(std::begin(armor.pts), std::end(armor.pts));
//   cv::Rect bbox = cv::boundingRect(pts_vec);



//   int new_width = static_cast<int>(bbox.width * expand_ratio_w_);
//   int new_height = static_cast<int>(bbox.height * expand_ratio_h_);
//   int new_x = static_cast<int>(bbox.x - (new_width - bbox.width) / 2);
//   int new_y = static_cast<int>(bbox.y - (new_height - bbox.height) / 2);

//   // 保证不越界
//   new_x = std::max(new_x, 0);
//   new_y = std::max(new_y, 0);

//   if (new_x + new_width > src.cols) new_width = src.cols - new_x;
//   if (new_y + new_height > src.rows) new_height = src.rows - new_y;

//   armor.new_x = new_x;
//   armor.new_y = new_y;

//   cv::Rect expanded_rect(new_x, new_y, new_width, new_height);
//   cv::Mat litroi = src(expanded_rect).clone();
//   cv::Mat litroi_color = src(expanded_rect).clone();
//   cv::cvtColor(litroi, litroi, cv::COLOR_RGB2GRAY);


//   cv::Mat  litroi_gray = litroi.clone();
//   armor.whole_gray_img = litroi_gray;

//   cv::threshold(litroi, litroi, binary_thres_, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);


//   // === 装甲板透视变换 ===
//   cv::Point2f lights_vertices[4] = {armor.pts[0], armor.pts[1], armor.pts[2], armor.pts[3]};

//   const int top_light_y = (warp_height - light_length) / 2 - 1;
//   const int bottom_light_y = top_light_y + light_length;
//   const int warp_width = is_large ? large_armor_width : small_armor_width;

//   cv::Point2f target_vertices[4] = {
//     cv::Point(0, bottom_light_y),
//     cv::Point(0, top_light_y),
//     cv::Point(warp_width - 1, top_light_y),
//     cv::Point(warp_width - 1, bottom_light_y),
//   };

//   cv::Mat number_image;
//   auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
//   cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));


  

//   // 获取 ROI（中心字符区域）
//   number_image = number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

//   // 灰度 + 二值化
//   cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY);
//   cv::threshold(number_image, number_image,0 , 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

//   // 上下翻转
//   cv::Mat flipped_image;
//   cv::flip(number_image, flipped_image, 0);
  

//   armor.number_img = flipped_image;
//   armor.whole_binary_img = litroi;
//   armor.whole_rgb_img = litroi_color;

//   cv::imshow("number_image", flipped_image);
//   cv::waitKey(1);
  
  
// }
void OpenVino::extractNumberImage(const cv::Mat & src, ArmorObject & armor)  {
  // Light length in image
  static const int light_length = 12;
  // Image size after warp
  static const int warp_height = 28;
  static const int small_armor_width = 32;
  static const int large_armor_width = 54;
  // Number ROI size
  static const cv::Size roi_size(20, 28);
  static const cv::Size input_size(28, 28);

  std::vector<cv::Point2f> pts_vec(std::begin(armor.pts), std::end(armor.pts));
  cv::Rect bbox = cv::boundingRect(pts_vec);

  float expand_ratio_w = 2.0f;
  float expand_ratio_h = 1.5f;
  int new_width = static_cast<int>(bbox.width * expand_ratio_w);
  int new_height = static_cast<int>(bbox.height * expand_ratio_h);
  int new_x = static_cast<int>(bbox.x - (new_width - bbox.width) / 2);
  int new_y = static_cast<int>(bbox.y - (new_height - bbox.height) / 2);

  // 保证不越界
  new_x = std::max(new_x, 0);
  new_y = std::max(new_y, 0);

  if (new_x + new_width > src.cols) new_width = src.cols - new_x;
  if (new_y + new_height > src.rows) new_height = src.rows - new_y;

  armor.new_x = new_x;
  armor.new_y = new_y;

  cv::Rect expanded_rect(new_x, new_y, new_width, new_height);
  cv::Mat litroi = src(expanded_rect).clone();
  cv::Mat litroi_color = src(expanded_rect).clone();
  cv::cvtColor(litroi, litroi, cv::COLOR_RGB2GRAY);


  cv::Mat  litroi_gray = litroi.clone();
  armor.whole_gray_img = litroi_gray;

  cv::threshold(litroi, litroi, binary_thres_, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  // Warp perspective transform
  cv::Point2f lights_vertices[4] = {
    armor.pts[0], armor.pts[1], armor.pts[2], armor.pts[3]};

  const int top_light_y = (warp_height - light_length) / 2 - 1;
  const int bottom_light_y = top_light_y + light_length;
  const int warp_width = (armor.number == ArmorNumber::NO1 || armor.number == ArmorNumber::BASE) ? small_armor_width : large_armor_width;
  cv::Point2f target_vertices[4] = {
    cv::Point(0, bottom_light_y),
    cv::Point(0, top_light_y),
    cv::Point(warp_width - 1, top_light_y),
    cv::Point(warp_width - 1, bottom_light_y),
  };
  cv::Mat number_image;
  auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
  cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

  // Get ROI
  number_image = number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

  // Binarize
  cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY);
  cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  cv::resize(number_image, number_image, input_size);
      cv::Mat flipped_image;
    cv::flip(number_image, flipped_image, 0);
  armor.number_img = flipped_image;
  armor.whole_binary_img = litroi;
  armor.whole_rgb_img = litroi_color;
  // cv::imshow("number_image",flipped_image);
  // cv::waitKey(1);
  return ;
}
  std::vector<Light> OpenVino::findLights(const cv::Mat &rgb_img,
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
      //armor.is_ok = !armor.lights.empty();
  
      return all_lights;
  }
  
  
  
  bool OpenVino::isLight(const Light &light) noexcept {
    // The ratio of light (short side / long side)
    float ratio = light.width / light.length;
    bool ratio_ok = light_params_.min_ratio < ratio && ratio < light_params_.max_ratio;
  
    bool angle_ok = light.tilt_angle < light_params_.max_angle;
  
    bool is_light = ratio_ok && angle_ok;
  
  
    return is_light;
  }
  void OpenVino::detect(ArmorObject & armor)
  { 
    lights_=findLights(armor.whole_rgb_img,armor.whole_binary_img,armor);
   
    LightCornerCorrector corner_corrector;
    corner_corrector.correctCorners(armor);
    
  }
//   bool OpenVino::classifyNumber(ArmorObject & armor)
// {
//   static thread_local std::unique_ptr<cv::dnn::Net> thread_net;

//   if (!thread_net) {
//     thread_net = std::make_unique<cv::dnn::Net>(cv::dnn::readNetFromONNX(classify_model_path_));
//     if (thread_net->empty()) {
//       std::cerr << "Failed to load thread-local number classifier model." << std::endl;
//       return false;
//     }
//   }

//   cv::Mat image = armor.number_img.clone();
//   image = image / 255.0;

//   cv::Mat blob;
//   cv::dnn::blobFromImage(image, blob);

//   thread_net->setInput(blob);  // ✅ 正确使用线程局部变量
//   cv::Mat outputs = thread_net->forward();

//   float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());
//   cv::Mat softmax_prob;
//   cv::exp(outputs - max_prob, softmax_prob);
//   float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
//   softmax_prob /= sum;

//   double confidence;
//   cv::Point class_id_point;
//   cv::minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
//   int label_id = class_id_point.x;

//   armor.confidence = confidence;

//   static const std::map<int, ArmorNumber> label_to_armor_number = {
//     {0, ArmorNumber::NO1}, {1, ArmorNumber::NO2}, {2, ArmorNumber::NO3},
//     {3, ArmorNumber::NO4}, {4, ArmorNumber::NO5}, {5, ArmorNumber::OUTPOST},
//     {6, ArmorNumber::SENTRY}, {7, ArmorNumber::BASE}
//   };

//   if (label_id < 8 && label_to_armor_number.find(label_id) != label_to_armor_number.end()) {
//     armor.number = label_to_armor_number.at(label_id);
//     return true;
//   } else {
//     armor.confidence = 0;
//     return false;
//   }
// }
bool OpenVino::classifyNumber(ArmorObject & armor)  {
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

void OpenVino::setCallback(DetectorCallback callback) { infer_callback_ = callback; }
bool OpenVino::processCallback(
    const cv::Mat resized_img, Eigen::Matrix3f transform_matrix, int64_t timestamp_nanosec,
    const cv::Mat & src_img)
  {
    // BGR->RGB, u8(0-255)->f32(0.0-1.0), HWC->NCHW
    // note: TUP's model no need to normalize
   // auto start =std::chrono::high_resolution_clock::now();
    cv::Mat blob =
      cv::dnn::blobFromImage(resized_img, 1., cv::Size(INPUT_W, INPUT_H), cv::Scalar(0, 0, 0), true);
  
    // Feed blob into input
    auto input_port = compiled_model_->input();
    ov::Tensor input_tensor(
      input_port.get_element_type(), ov::Shape(std::vector<size_t>{1, 3, INPUT_W, INPUT_H}),
      blob.ptr(0));
  
    // Start inference
    auto infer_request = compiled_model_->create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    // infer_request.start_async();
    // infer_request.wait();
  
    auto output = infer_request.get_output_tensor();
  
    // Process output data
    auto output_shape = output.get_shape();
    // 3549 x 21 Matrix
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());
  
    // Parsed variable
    std::vector<ArmorObject> objs_tmp, objs_result;
    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    std::vector<int> indices;
  
    // Parse YOLO output
    generate_proposals(
      objs_tmp, scores, rects, output_buffer, transform_matrix, this->conf_threshold_,
      this->grid_strides_);
  
    // TopK
    std::sort(objs_tmp.begin(), objs_tmp.end(), [](const ArmorObject & a, const ArmorObject & b) {
      return a.prob > b.prob;
    });
    if (objs_tmp.size() > static_cast<size_t>(this->top_k_)) {
      objs_tmp.resize(this->top_k_);
    }
  
    nms_merge_sorted_bboxes(objs_tmp, indices, this->nms_threshold_);
  
    for (size_t i = 0; i < indices.size(); i++) {
      objs_result.push_back(std::move(objs_tmp[indices[i]]));
  
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
    // auto end =std::chrono::high_resolution_clock::now();
    // WUST_INFO("openvino") << "infer time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms" ;
  
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
      
      extractNumberImage(src_img, armor);
      
      classifyNumber(armor);
     
      detect(armor);
    
    }
  
    // NMS & TopK
    // cv::dnn::NMSBoxes(
    //   rects, scores, this->conf_threshold_, this->nms_threshold_, indices, 1.0,
    //   this->top_k_);
    // for (size_t i = 0; i < indices.size(); ++i) {
    //   objs_result.push_back(std::move(objs_tmp[i]));
    // }
    
    // Call callback function
    if (this->infer_callback_) {
      this->infer_callback_(objs_result, timestamp_nanosec, src_img);
      return true;
    }
   
    return false;
  }
  void OpenVino::pushInput(const cv::Mat& rgb_img, int64_t timestamp_nanosec) {
    if (rgb_img.empty()) {
      return;
    }
  
  
  
    Eigen::Matrix3f transform_matrix;
    cv::Mat resized_img = letterbox(rgb_img, transform_matrix);
  
  
  
    thread_pool_->enqueue([this, resized_img, transform_matrix, timestamp_nanosec, rgb_img]() {
      this->processCallback(resized_img, transform_matrix, timestamp_nanosec, rgb_img);
      
    });
  }

