#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

struct Vector3 {
  double x;
  double y;
  double z;
};

struct TwistData {
  pthread_mutex_t mutex;
  Vector3 linear;
  Vector3 angular;
};
// #define SHM_NAME "/twist_shared_memory"
// #define SHM_SIZE sizeof(TwistData)

// inline TwistData* init_twist_shared_memory(bool create) {
//     int shm_fd;
//     if (create) {
//         shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
//     } else {
//         shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
//     }

//     if (shm_fd == -1) {
//         perror("shm_open");
//         return nullptr;
//     }

//     if (create) {
//         if (ftruncate(shm_fd, SHM_SIZE) == -1) {
//             perror("ftruncate");
//             return nullptr;
//         }
//     }

//     void* ptr = mmap(0, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd,
//     0); if (ptr == MAP_FAILED) {
//         perror("mmap");
//         return nullptr;
//     }

//     return static_cast<TwistData*>(ptr);
// }
