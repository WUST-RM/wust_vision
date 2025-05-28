#include "type/type.hpp"
double R_x = 0.0;
double R_y = 0.0;
double R_z = 0.0;
double R_yaw = 0.0;
double time_ = 0.0;

double s2qx=0.0;
double s2qy=0.0;
double s2qz=0.0;
double s2qyaw=0.0;


double s2qx_min = 0.1;
double s2qx_max = 100.0;
double s2qy_min = 0.1;
double s2qy_max = 100.0;
double s2qz_min = 0.1;
double s2qz_max = 100.0;
double s2qyaw_min = 0.1;
double s2qyaw_max = 100.0;
double last_x_ = 0.0, last_y_ = 0.0, last_z_ = 0.0, last_yaw_ = 0.0;
std::vector<Armors> datas;
std::chrono::steady_clock::time_point last_time_ = std::chrono::steady_clock::now();


double orientationToYaw(const tf2::Quaternion& orientation) {
    tf2::Quaternion q(orientation.x, orientation.y, orientation.z, orientation.w);
    tf2::Matrix3x3 m(q);
    double yaw, pitch, roll;
    m.getRPY(roll, pitch, yaw);
    return yaw;
  }
  
  
  void ex(double& a,double& min,double& max)
  {
    if(a<min)
    {
        min = a;
    }
    if(a>max)
    {
        max = a;
    }
        
    
  
  }
  void command_callback(Armors& armors)
  {
    std::cout<<"已经收集了"<<datas.size()<<"个数据"<<std::endl;
          
          datas.push_back(armors);
          auto current_time = std::chrono::steady_clock::now();
          auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time_).count()/1000.0;
          //last_time_ = current_time;
  
          if (!armors.armors.empty()&&delta_time>0.5) {
              last_time_ = current_time;
              if(datas.size()==1)
              {
                  last_x_ = armors.armors[0].pos.x;
                  last_y_ = armors.armors[0].pos.y;
                  last_z_ = armors.armors[0].pos.z;
                  last_yaw_ = orientationToYaw(armors.armors[0].ori);
  
              }
              else{
  
                      double current_x = armors.armors[0].pos.x;
                      double current_y = armors.armors[0].pos.y;
                      double current_z = armors.armors[0].pos.z;
                      double current_yaw = orientationToYaw(armors.armors[0].ori);
  
                      if (delta_time > 0) {
                          double v_x = (current_x - last_x_) / delta_time;
                          double v_y = (current_y - last_y_) / delta_time;
                          double v_z = (current_z - last_z_) / delta_time;
                          double v_yaw = (current_yaw - last_yaw_) / delta_time;
  
                          // 调整Q的计算逻辑，例如速度增大时Q增大
                          s2qx = exp(-(abs(v_x) + 0.5*abs(v_yaw))) * (s2qx_max - s2qx_min) + s2qx_min;
                          ex(s2qx,s2qx_min,s2qx_max);
                          s2qy = exp(-(abs(v_y) + 0.5*abs(v_yaw))) * (s2qy_max - s2qy_min) + s2qy_min;
                          ex(s2qy,s2qy_min,s2qy_max);
                          s2qz = exp(-(abs(v_z) + 0.5*abs(v_yaw))) * (s2qz_max - s2qz_min) + s2qz_min;
                          ex(s2qz,s2qz_min,s2qz_max);
                          s2qyaw = exp(-(abs(v_x) + 0.5*abs(v_z))) * (s2qyaw_max - s2qyaw_min) + s2qyaw_min;
                          ex(s2qyaw,s2qyaw_min,s2qyaw_max);
                      }
  
                      // 更新前一次状态
                      last_x_ = current_x;
                      last_y_ = current_y;
                      last_z_ = current_z;
                      last_yaw_ = current_yaw;
                  }
          }
  
  
          if(datas.size()==5000)
          {
              double all_x = 0.0;
              double all_y = 0.0;
              double all_z = 0.0;
              double all_yaw = 0.0;
      
            for (int i = 0; i < 5000; i++)
              {
                  std::cout << i << std::endl;
  
                  // 检查 datas[i].armors 是否为空
                  if (!datas[i].armors.empty())
                  {
                      time_ = time_ +1;
                      all_x += datas[i].armors[0].pos.x;
                      all_y += datas[i].armors[0].pos.y;
                      all_z += datas[i].armors[0].pos.z;
                      all_yaw += orientationToYaw(datas[i].armors[0].ori);
  
                  }
                  else
                  {
                      
                  }
              }
              double mean_x = all_x/time_;
              double mean_y= all_y/time_;
              double mean_z = all_z/time_;
              double mean_yaw = all_yaw/time_;
              
  
              double variance_x = 0.0;
              double variance_y = 0.0;
              double variance_z = 0.0;
              double variance_yaw = 0.0;
             
              for(int i = 0 ;i<5000;i++)
              {
                   if (!datas[i].armors.empty())
                   {
                   // 修改这里：使用 datas[i].armors[0] 访问第一个 Armor 对象
                      variance_x += pow(mean_x - datas[i].armors[0].pos.x, 2);
                      variance_y += pow(mean_y - datas[i].armors[0].pos.y, 2);
                      variance_z += pow(mean_z - datas[i].armors[0].pos.z, 2);
                      variance_yaw += pow(mean_yaw - orientationToYaw(datas[i].armors[0].ori), 2);
                   }
                  
              }
  
              R_x = variance_x/time_;
              R_y = variance_y/time_;
              R_z = variance_z/time_;
              R_yaw = variance_yaw/time_;
              
  
          }
  
          if(datas.size()>5000)
          {
               // 设置输出格式为固定小数点，并设置小数点后的精度
              std::cout << std::fixed << std::setprecision(10);
              std::cout<<"R_x:"<<R_z<<std::endl;
              std::cout<<"R_y:"<<R_x<<std::endl;
              std::cout<<"R_z:"<<R_y<<std::endl;
              std::cout<<"R_yaw:"<<R_yaw<<std::endl;
  
              std::cout<<"s2qx:"<<s2qz<<std::endl;
              std::cout<<"s2qy:"<<s2qx<<std::endl;
              std::cout<<"s2qz:"<<s2qy<<std::endl;
              std::cout<<"s2qyaw:"<<s2qyaw<<std::endl;
          }
  }