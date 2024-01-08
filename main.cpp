#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

struct Sphere {
    vector<float> center;
    array<int, 3> color{};
    float radius{};
    int specular{};
    float reflective{};
};

struct Light {
    int type{}; // 0 == ambient, 1 == point, 2 == directional
    vector<float> center;
    float intensity{};
};

#define Vw 1.0
#define Vh 1.0
#define Cw 600.0
#define Ch 600.0
#define d 1.0 // distance


class RayTracing {
public: // Access specifier
    vector<Sphere> spheres;
    vector<Light> allOfTheLights;
    vector<float> origin = {0, 0, 0};
    explicit RayTracing(vector<Sphere> s, vector<Light> lights){
        spheres = std::move(s);
        allOfTheLights = std::move(lights);
    }

    static vector<float> addVectors(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        if (vec1.size() != vec2.size()) {
            // Handling unequal vector sizes, you might want to throw an exception or handle this differently
            std::cerr << "Vectors have different sizes!" << std::endl;
            return {}; // Return an empty vector
        }

        std::vector<float> result;
        result.reserve(vec1.size()); // Reserve space for the result vector

        // Subtract corresponding elements
        for (size_t i = 0; i < vec1.size(); ++i) {
            result.push_back(vec1[i] + vec2[i]);
        }

        return result;
    }

    static vector<float> subtractVectors(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        if (vec1.size() != vec2.size()) {
            // Handling unequal vector sizes, you might want to throw an exception or handle this differently
            std::cerr << "Vectors have different sizes!" << std::endl;
            return {}; // Return an empty vector
        }

        std::vector<float> result;
        result.reserve(vec1.size()); // Reserve space for the result vector

        // Subtract corresponding elements
        for (size_t i = 0; i < vec1.size(); ++i) {
            result.push_back(vec1[i] - vec2[i]);
        }

        return result;
    }

    static vector<float> multiplyVectors(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        if (vec1.size() != vec2.size()) {
            // Handling unequal vector sizes, you might want to throw an exception or handle this differently
            std::cerr << "Vectors have different sizes!" << std::endl;
            return {}; // Return an empty vector
        }

        std::vector<float> result;
        result.reserve(vec1.size()); // Reserve space for the result vector

        // Dividing corresponding elements
        for (size_t i = 0; i < vec1.size(); ++i) {
            result.push_back(vec1[i] * vec2[i]);
        }
        return result;
    }

    static vector<float> divideVectors(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        if (vec1.size() != vec2.size()) {
            // Handling unequal vector sizes, you might want to throw an exception or handle this differently
            std::cerr << "Vectors have different sizes!" << std::endl;
            return {}; // Return an empty vector
        }

        std::vector<float> result;
        result.reserve(vec1.size()); // Reserve space for the result vector

        // Dividing corresponding elements
        for (size_t i = 0; i < vec1.size(); ++i) {
            result.push_back(vec1[i] / vec2[i]);
        }
        return result;
    }

    static float betrag(const std::vector<float>& vec) {
        return sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
    }

    static float dot(vector<float> a, vector<float> b){
        return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
    }

    static vector<float> canvasToViewport(float x, float y) {
        y = -y;
        return {static_cast<float>(x*Vw/Cw), static_cast<float>(y*Vh/Ch), d};
    }


    static array<float, 2> mitternacht(float a, float b, float c){
        array<float, 2> ret{};
        float underSqrt = b*b - 4*a*c;
        if(underSqrt < 0)
            throw runtime_error("No solution");
        float tmp =  sqrt(underSqrt);
        ret[0] =  (-b + tmp)/(2*a);
        ret[1] = (-b - tmp)/(2*a);
        return ret;
    }

    array<float, 2> closestIntersection(const vector<float>& o, const vector<float>& vec, float tFrom, float tTo){
        vector<array<float, 2>> distancesT;
        distancesT.reserve(spheres.size());
        float a = dot(vec, vec);
        for(int i = 0; i < spheres.size(); i++){
            vector<float> co = subtractVectors(o, spheres[i].center);
            float b = 2 * dot(co, vec);
            float c = dot(co, co) - spheres[i].radius*spheres[i].radius;
            try{
                distancesT.push_back(mitternacht(a,b,c));
            } catch(const runtime_error &error ){
                distancesT.push_back({INFINITY, INFINITY});
            }
        }
        float closest[2] = {INFINITY, -1};
        for(int i = 0; i < spheres.size(); i++){
            if(distancesT[i][0] < closest[0] && distancesT[i][0] > tFrom && distancesT[i][0] < tTo){
                closest[0] = distancesT[i][0];
                closest[1] = (float) i;
            }
            if(distancesT[i][1] < closest[0] && distancesT[i][1] > tFrom && distancesT[i][1] < tTo){
                closest[0] = distancesT[i][1];
                closest[1] = (float) i;
            }
        }
        return reinterpret_cast<array<float, 2> &&>(closest);
    }


    array<unsigned char, 3> TraceRay(const vector<float>& o, const vector<float>& vec, float tFrom, float tTo, int recursion_depth){
        array<float, 2> closest = closestIntersection(o,vec, tFrom, tTo);
        if(closest[1] == -1)
            return {0, 0, 0};
       vector<float> intersection = addVectors(origin,multiplyVectors(vec, {closest[0], closest[0], closest[0]}));
       float light = lighting(intersection, spheres[(int)closest[1]], multiplyVectors(vec, {-1, -1, -1}));
       array<unsigned char, 3> localLight = {static_cast<unsigned char>(spheres[(int)closest[1]].color[0]*light), static_cast<unsigned char>(spheres[(int)closest[1]].color[1]*light), static_cast<unsigned char>(spheres[(int)closest[1]].color[2]*light)};
       float curReflect = spheres[closest[1]].reflective;
       if(curReflect <= 0 || recursion_depth <= 0){
           return localLight;
       }
       vector<float> reflect = reflectRay(multiplyVectors(vec, {-1, -1, -1}), calcShpereNormal(spheres[(int)closest[1]],intersection));
       array<unsigned char, 3> reflectedColor = TraceRay(intersection, reflect, 0.001, INFINITY, recursion_depth-1);
       array<unsigned char, 3> mulLight = {static_cast<unsigned char>(localLight[0]*(1-curReflect) + reflectedColor[0]*curReflect),static_cast<unsigned char>(localLight[1]*(1-curReflect) + reflectedColor[1]*curReflect), static_cast<unsigned char>(localLight[2]*(1-curReflect) + reflectedColor[2]*curReflect)};
       return mulLight;
    }

    static vector<float> calcShpereNormal(const Sphere& s,const vector<float>& point){
        vector<float> vecDiff = subtractVectors(point, s.center);
        float betragValue = betrag(vecDiff);
        vector<float> betragVec = {betragValue, betragValue, betragValue};
        return divideVectors(vecDiff, betragVec);
    }

    static vector<float> reflectRay(const vector<float>& a, const vector<float>& b){
        float dotProductOfNL = dot(a, b)*2;
        vector<float> rVec = multiplyVectors({dotProductOfNL,dotProductOfNL,dotProductOfNL}, a);
        return subtractVectors(rVec, b);
    }

    float lighting(const vector<float>& hitPoint, const Sphere& hitSphere, const vector<float>& viewVec){
        vector<float> normal = calcShpereNormal(hitSphere, hitPoint);
        float betragVonNormal = betrag(normal);
        float viewBetrag = betrag(viewVec);
        float totalIntensity = 0;
        float areaRatio = 0;
        float dotProductOfNL;
        double reflectionSum;
        float shiny;
        vector<float> rVec;
        vector<float> lightVector;
        for(int i = 0; i < allOfTheLights.size(); i++){
            switch(allOfTheLights[i].type){
                case 0: // ambient
                    totalIntensity += allOfTheLights[i].intensity;
                    break;
                case 1: // point
                    lightVector = subtractVectors(allOfTheLights[i].center, hitPoint);
                    if((int)closestIntersection(hitPoint,lightVector,0.001, INFINITY)[1] != -1)
                        break;
                    if(hitSphere.specular > 0){
                        rVec = reflectRay(normal, lightVector);
                        shiny = dot(rVec, viewVec);
                        if(shiny < 0){
                            reflectionSum = 0;
                        } else{
                            reflectionSum = pow(shiny / (betrag(rVec) * viewBetrag), hitSphere.specular);
                        }
                    } else reflectionSum = 0;
                    areaRatio = dot(normal, lightVector);
                    if(areaRatio < 0){
                        break;
                    }
                    else
                        totalIntensity += allOfTheLights[i].intensity * ((areaRatio / (betrag(lightVector)*betragVonNormal)) + reflectionSum);
                    break;
                case 2: // direction
                    if((int)closestIntersection(hitPoint,lightVector,0.001, INFINITY)[1] != -1)
                        break;
                    if(hitSphere.specular > 0){
                        rVec = reflectRay(normal, allOfTheLights[i].center);
                        shiny = dot(rVec, viewVec);
                        if(shiny < 0){
                            reflectionSum = 0;
                        }
                        else
                            reflectionSum = pow(shiny / (betrag(rVec) * viewBetrag), hitSphere.specular);
                    } else reflectionSum = 0;
                    areaRatio = dot(normal, allOfTheLights[i].center);
                    if(areaRatio < 0){
                        break;
                    }
                    else
                        totalIntensity += allOfTheLights[i].intensity * ((areaRatio / (betrag(allOfTheLights[i].center)*betragVonNormal)) + reflectionSum);
                    break;
            }
        }
       // cout << totalIntensity << endl;
       if(totalIntensity > 1)
           totalIntensity = 1;
        return totalIntensity;
    }

private: // Private access specifier

};




int main() {
    // Read an image
    cv::VideoWriter video("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(600, 600));
    //cv::Mat image = cv::imread(R"(C:\Users\gusta\CLionProjects\rayTracing\image.jpg)");

    if (!video.isOpened()) {
        cout << "Could not open the output video file for write" << endl;
        return -1;
    }

    float projection_plane_d = 1;
    Sphere s1;
    s1.center = {0, -1, 3};
    s1.color = {0, 0, 255};
    s1.radius = 1;
    s1.specular = 500;
    s1.reflective = 0.2;
    Sphere s2;
    s2.center = {2, 0, 4};
    s2.color = {255, 0, 0};
    s2.radius = 1;
    s2.specular = 500;
    s2.reflective = 0.3;
    Sphere s3;
    s3.center = {-2, 0, 4};
    s3.color = {0, 255, 0};
    s3.radius = 1;
    s3.specular = 10;
    s3.reflective = 0.4;
    Sphere s4;
    s4.color = {0, 255, 255};
    s4.center = {0, -8001, 10};
    s4.radius = 8000;
    s4.specular = 20;
    s4.reflective = 0.5;
    vector<Sphere> objects = {s1, s2, s3};

    Light l1;
    //l1.center = {0, -1, 3};
    l1.type = 0;
    l1.intensity = 0.2;
    Light l2;
    l2.center = {2, 1, 0};
    l2.type = 1;
    l2.intensity = 0.6;
    Light l3;
    l3.center = {1, 4, 4};
    l3.type = 2;
    l3.intensity = 0.2;
    vector<Light> lights = {l1, l2, l3};

    RayTracing tracer(objects, lights);
    int frameCount = 10;
    for(int i = 0; i < frameCount; i++){
        cv::Mat frame(600, 600, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int y = (int)(-Ch/2); y < (int)(Ch/2); y ++) {
            for (int x = (int)(-Cw/2); x < (int)(Cw/2); x ++) {
                // Set pixel color (BGR format)
                array<unsigned char, 3> color = tracer.TraceRay(tracer.origin, RayTracing::canvasToViewport((float)x,(float)y), 1, INFINITY, 3);
                frame.at<cv::Vec3b>(y+(int)Ch/2, x+(int)Cw/2) = {color[0], color[1], color[2]};// Green color (BGR: Blue, Green, Red)
                // cout << tracer.TraceRay(RayTracing::canvasToViewport(x,y), 1, 10000000) << endl;
            }
        }
        cv::imshow("Cur frame", frame);
        cv::waitKey();
        video.write(frame);
        tracer.allOfTheLights[1].center[2] += 0.3;
        tracer.allOfTheLights[1].center[0] -= 0.3;
        if(i < frameCount/2)
            tracer.spheres[0].center[1] += 0.3;
        else tracer.spheres[0].center[1] -= 0.3;
        //tracer.spheres[0].center[2] += 0.4;
        cout << i+1 << "/" << frameCount << endl;
    }


    // Display the modified image
    //cv::imshow("Modified Image", image);
    //cv::waitKey(0);
    cout << "render complete" << endl;
    video.release();
    return 0;
}