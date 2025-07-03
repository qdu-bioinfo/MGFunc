// Updated at Feb. 2, 2025
// Updated by Yu Zhang
// Bioinformatics Group, College of Computer Science & Technology, Qingdao University
// version 1.0

#ifndef TIMER_H
#define TIMER_H
#include <sys/time.h>
struct my_timer
{
    timeval ts, te; //time start and time end
    float dt; // time distance(ms)
    void start(){
        gettimeofday(&ts, NULL);
    }
    void stop(){
        gettimeofday(&te, NULL);
        long int dt_sec  = te.tv_sec - ts.tv_sec;
        long int dt_usec = te.tv_usec - ts.tv_usec;
        dt = dt_sec * 1.0e3 + dt_usec / 1.0e3;
    }
};
#endif
