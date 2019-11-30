#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>

using namespace std;

struct vec3
{
    float x;
    float y;
    float z;

    vec3(float x, float y, float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    vec3()
    {
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }

    float length()
    {
        return sqrtf(x * x + y * y + z * z);
    }

    void normalise()
    {
        x = x / length();
        y = y / length();
        z = z / length();
    }

    vec3 operator-(vec3 &other)
    {
        return vec3(this->x - other.x, this->y - other.y, this->z - other.z);
    }

    // vec3 operator*(float &scalar)
    // {
    //     return vec3(this->x * scalar, this->y * scalar, this->z * scalar);
    // }

    vec3 operator*(float scalar){
        return vec3(this->x * scalar, this->y * scalar, this->z * scalar);
    }

    vec3 operator+(vec3 other)
    {
        return vec3(this->x + other.x, this->y + other.y, this->z + other.z);
    }

    vec3 operator/(float scalar)
    {
        // cout << "divided " << this->z << endl;
        return vec3(this->x / scalar, this->y / scalar, this->z / scalar);
    }

    vec3 operator+(float other){
        return vec3(this->x + other, this->y + other, this->z + other);
    }
};

vec3 cross(vec3 &A, vec3 &B)
{
    float x = (A.y * B.z) - (A.z * B.y);
    float y = (A.z * B.x) - (A.x * B.z);
    float z = (A.x * B.y) - (A.y * B.x);
    float norm = sqrtf(pow(x, 2) + pow(y, 2) + pow(z, 2));
    return vec3(x, y, z);
};

float dot(vec3 &A, vec3 &B)
{
    return (A.x * B.x) + (A.y * B.y) + (A.z * B.z);
}

struct eye
{
    vec3 position = vec3(0.0, 0.0, 0.0);
    vec3 direction = vec3(0.0, 0.0, 1.0);
    vec3 up = vec3(0.0, 1.0, 0.0);
    float fov = 90.0;

    eye(vec3 position, vec3 direction, vec3 up, float fov)
    {
        this->position = position;
        this->direction = direction;
        this->up = up;
        this->fov = fov;
    }
};

struct light
{
    vec3 position;
    vec3 direction;
    float diffuseIntensity;
    float specularIntensity;
    float ambientIntensity;
};

struct Material
{
    float diffuseIntensity;
    float specularIntensity;
    float ambientIntensity;
};

struct triangle
{
    vec3 A = vec3(61, 10, 1);
    vec3 B = vec3(100, 100, 1);
    vec3 C = vec3(25, 90, 1);

    Material m;

    triangle(vec3 A, vec3 B, vec3 C, Material m)
    {
        this->A = A;
        this->B = B;
        this->C = C;
        this->m = m;
    }
};

struct ray
{
    vec3 start;
    vec3 direction;

    ray(vec3 start, vec3 direction)
    {
        this->start = start;
        this->direction = direction;
    }
};

vec3 getPlaneNormal(triangle t)
{
    vec3 planeDim1 = t.B - t.A;
    vec3 planeDim2 = t.C - t.A;
    vec3 normal = cross(planeDim1, planeDim2);
    normal.normalise();
    return normal;
}

bool isInsideTriangle(triangle t, vec3 point, vec3 planeNormal)
{
    vec3 triNormal1;
    vec3 triNormal2;
    vec3 triNormal3;

    vec3 edge0 = t.B - t.A;
    vec3 edge1 = t.C - t.B;
    vec3 edge2 = t.A - t.C;

    vec3 p0 = point - t.A;
    vec3 p1 = point - t.B;
    vec3 p2 = point - t.C;

    triNormal1 = cross(edge0, p0);
    triNormal2 = cross(edge1, p1);
    triNormal3 = cross(edge2, p2);

    float lTriNormal1 = triNormal1.length();
    float lTriNormal2 = triNormal2.length();
    float lTriNormal3 = triNormal3.length();

    float dot1 = dot(planeNormal, triNormal1);
    if (dot1 < 0)
        return false;
    float dot2 = dot(planeNormal, triNormal2);
    if (dot2 < 0)
        return false;
    float dot3 = dot(planeNormal, triNormal3);
    if (dot3 < 0)
        return false;

    return true;
}

bool isIntersectingTriangle(ray r, triangle t, vec3 &pointOut)
{
    vec3 normal = getPlaneNormal(t);
    // cout << normal.x << " " << normal.y << " " << normal.z << endl;
    //float distanceToPlane = dot(normal, t.A);
    //float param = (dot(normal, r.start) + distanceToPlane) / dot(normal, r.direction);
    float denom = dot(normal, r.direction);

    if(denom == 0){
        return false;
    }
    
    vec3 numerator = t.A - r.start;
    float d = dot(numerator, normal) / denom ;


    vec3 p = r.start + (r.direction * d);
    // cout << p.x << " " << p.y << " " << p.z << endl;
    if (isInsideTriangle(t, p, normal))
    {
        pointOut = p;

        return true;
    }
    return false;
}



float distance(vec3 point, vec3 vertex1, vec3 vertex2)
{
    //point in relation to one end of line
    float x = point.x - vertex1.x;
    float y = point.y - vertex1.y;
    //vector on line
    float ex = vertex2.x - vertex1.x;
    float ey = vertex2.y - vertex1.y;
    //normal to line
    float invex = -ey;
    float invey = ex;
    //dot product of point vector and normal
    float dotproduct = (x * invex) + (y * invey);
    //magnitude of normal
    float magnitudeNormal = sqrt((invex * invex) + (invey * invey));
    //removing scaling from distance
    return dotproduct / magnitudeNormal;
}

void setupImage(vector<vector<int>> &image, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            image[i][j * 3] = 255;
            image[i][j * 3 + 1] = 255;
            image[i][j * 3 + 2] = 192;
        }
    }
}

void baryinterp(int &R, int &G, int &B, vec3 point, triangle t)
{
    R = 0;
    G = 0;
    B = 0;
    float alpha = distance(point, t.B, t.C) / distance(t.A, t.B, t.C); //distance from xstep,ystep to CB
    float beta = distance(point, t.A, t.C) / distance(t.B, t.A, t.C);  //distance from xstep,ystep to AC
    float gamma = distance(point, t.B, t.A) / distance(t.C, t.B, t.A); //distance from xstep,ystep to BA
    R = 255 * alpha;
    G = 255 * beta;
    B = 255 * gamma;
    R = (R < 0) ? 0 : (R > 255) ? 255 : R;
    G = (G < 0) ? 0 : (G > 255) ? 255 : G;
    B = (B < 0) ? 0 : (B > 255) ? 255 : B;
};

ray castray(vec3 origin, vec3 location)
{
    // cout << "pixel x: " << x << " pixel y: " << y << endl;
    // float xr = (2 * ((location.x + 0.5) / width) - 1);
    // float yr = ((2 * ((location.y + 0.5) / height)) - 1);
    // cout << "worldspace x: " << xr << " worldspace y: " << yr << endl;
    // vec3 direction = vec3(xr, yr, location.z);
    vec3 direction = location - origin;
    //direction.normalise();
    return ray(origin, direction);
}

vec3 convertCoordinates(vec3 coord, int width, int height)
{
    float aspectRatio = float(width) / float(height);
    float xr = ((2 * ((coord.x) / width)) - 1) * aspectRatio;
    float yr = ((2 * ((coord.y) / height)) -1) * aspectRatio;
    return vec3(xr, yr, coord.z);
}


bool isInShadow(vec3 point, vec3 normal, triangle t, light l)
{
    triangle tWorld = triangle(convertCoordinates(t.A, 128, 128), convertCoordinates(t.B, 128, 128), convertCoordinates(t.C, 128, 128), t.m);
    ray r = castray(point + 0.01, l.position);
    vec3 pos;
    cout << r.direction.x << " " << r.direction.y << " " << r.direction.z << endl;
    if (isIntersectingTriangle(r, tWorld, pos))
    {
        cout << "yes" << endl;
        return true;
    }
    return false;
}


void drawImageHalfPlaneTest(vector<vector<int>> &image, eye e, triangle t, float xmax, float ymax)
{
    triangle tWorld = triangle(convertCoordinates(t.A, 128, 128), convertCoordinates(t.B, 128, 128), convertCoordinates(t.C, 128, 128), t.m);
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, tWorld, pointInTriangle))
            {
                image[ystep][xstep * 3] = 0;
                image[ystep][xstep * 3 + 1] = 0;
                image[ystep][xstep * 3 + 2] = 0;
            }
        }
    }
}

void drawImage(vector<vector<int>> &image, eye e, triangle t, float xmax, float ymax)
{
    triangle tWorld = triangle(convertCoordinates(t.A, 128, 128), convertCoordinates(t.B, 128, 128), convertCoordinates(t.C, 128, 128), t.m);
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, tWorld, pointInTriangle))
            {
                int R, G, B;
                baryinterp(R, G, B, pointInTriangle, tWorld);
                image[ystep][xstep * 3] = R;
                image[ystep][xstep * 3 + 1] = G;
                image[ystep][xstep * 3 + 2] = B;
            }
        }
    }
}

float computeDiffuse(vec3 point, light l, triangle t)
{
    vec3 triangleNormal = getPlaneNormal(t);
    vec3 vl = point - l.position;
    float numerator = dot(triangleNormal, vl);
    float denom = triangleNormal.length() * vl.length();
    // cout << normalised << endl;
    return l.diffuseIntensity * t.m.diffuseIntensity * (numerator / denom);
}

float computeSpecular(vec3 point, light l, triangle t, eye e)
{
    vec3 triangleNormal = getPlaneNormal(t);
    vec3 vl = l.position - point;
    vec3 ve = e.position - point;
    vec3 vb = (vl + ve) / 2;
    // cout << vb.x << endl;
    float numerator = dot(triangleNormal, vb);
    float denom = triangleNormal.length() * vb.length();

    float angle = numerator / denom;
    // cout << vb.x << vb.y << vb.z << endl;
    return l.specularIntensity * t.m.specularIntensity * powf64(angle, 100);
}

float computeAmbient(vec3 point, light l, Material m)
{
    return l.ambientIntensity * m.ambientIntensity;
}

void drawImageAmbient(vector<vector<int>> &image, eye e, triangle t, light l, float xmax, float ymax)
{
    triangle tWorld = triangle(convertCoordinates(t.A, 128, 128), convertCoordinates(t.B, 128, 128), convertCoordinates(t.C, 128, 128), t.m);
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, tWorld, pointInTriangle))
            {
                int R = 0;
                int G = 255;
                int B = 0;
                //baryinterp(R, G, B, pointInTriangle, tWorld);
                float amt = computeAmbient(pointInTriangle, l, tWorld.m);
                image[ystep][xstep * 3] = R * amt;
                image[ystep][xstep * 3 + 1] = G * amt;
                image[ystep][xstep * 3 + 2] = B * amt;
            }
        }
    }
}

void drawImageSpecular(vector<vector<int>> &image, eye e, triangle t, light l, float xmax, float ymax)
{
    //light lWorld = light{convertCoordinates(l.position, 128, 128), l.direction, l.diffuseIntensity, l.specularIntensity, l.ambientIntensity};
    triangle tWorld = triangle(convertCoordinates(t.A, 128, 128), convertCoordinates(t.B, 128, 128), convertCoordinates(t.C, 128, 128), t.m);
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, tWorld, pointInTriangle))
            {
                int R = 0;
                int G = 255;
                int B = 0;
                //baryinterp(R, G, B, pointInTriangle, tWorld);
                float amt = computeSpecular(pointInTriangle, l, tWorld, e);
                image[ystep][xstep * 3] = R * amt;
                image[ystep][xstep * 3 + 1] = G * amt;
                image[ystep][xstep * 3 + 2] = B * amt;
            }
        }
    }
}

void drawImageDiffuse(vector<vector<int>> &image, eye e, triangle t, light l, float xmax, float ymax)
{
    triangle tWorld = triangle(convertCoordinates(t.A, 128, 128), convertCoordinates(t.B, 128, 128), convertCoordinates(t.C, 128, 128), t.m);
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, tWorld, pointInTriangle))
            {
                int R = 0;
                int G = 255;
                int B = 0;
                //baryinterp(R, G, B, pointInTriangle, tWorld);
                float amt = computeDiffuse(pointInTriangle, l, tWorld);
                image[ystep][xstep * 3] = R * amt;
                image[ystep][xstep * 3 + 1] = G * amt;
                image[ystep][xstep * 3 + 2] = B * amt;
            }
        }
    }
}

void drawImageWithLighting(vector<vector<int>> &image, eye e, triangle t, light l, float xmax, float ymax)
{
    triangle tWorld = triangle(convertCoordinates(t.A, 128, 128), convertCoordinates(t.B, 128, 128), convertCoordinates(t.C, 128, 128), t.m);
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, tWorld, pointInTriangle))
            {
                int R = 0;
                int G = 255;
                int B = 0;
                //baryinterp(R, G, B, pointInTriangle, tWorld);
                float a = computeAmbient(pointInTriangle, l, tWorld.m);
                float d = computeDiffuse(pointInTriangle, l, tWorld);
                float s = computeSpecular(pointInTriangle, l, tWorld, e);

                R = R * a + R * d + R * s;
                G = G * a + G * d + G * s;
                B = B * a + B * d + G * s;

                R = (R < 0) ? 0 : (R > 255) ? 255 : R;
                G = (G < 0) ? 0 : (G > 255) ? 255 : G;
                B = (B < 0) ? 0 : (B > 255) ? 255 : B;

                image[ystep][xstep * 3] = R;
                image[ystep][xstep * 3 + 1] = G;
                image[ystep][xstep * 3 + 2] = B;
            }
        }
    }
}

void drawTriangles(vector<vector<int>> &image, eye e, vector<triangle> tris, light l, float xmax, float ymax)
{
    light lWorld = light{convertCoordinates(l.position, 128, 128), l.direction, l.diffuseIntensity, l.specularIntensity, l.ambientIntensity};
    for(int i = 0; i < tris.size(); i++){
        triangle tWorld = triangle(convertCoordinates(tris[i].A, 128, 128), convertCoordinates(tris[i].B, 128, 128), convertCoordinates(tris[i].C, 128, 128), tris[i].m);
        for (int ystep = ymax - 1; ystep >= 0; ystep--)
        {
            for (int xstep = 0; xstep < xmax; xstep++)
            {
                ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
                vec3 pointInTriangle;
                vec3 normal = getPlaneNormal(tWorld);
                bool inShadow = false;
                if (isIntersectingTriangle(r, tWorld, pointInTriangle))
                {
                    for(int j = 0; j < tris.size(); j++){
                        if(j == i){
                            continue;
                        }
                        if(inShadow){
                            break;
                        }
                        inShadow = isInShadow(pointInTriangle, normal, tris[j], l);
                    }
                    if(inShadow){ 
                        image[ystep][xstep * 3] = 0;
                        image[ystep][xstep * 3+1] = 0;
                        image[ystep][xstep * 3+2] = 0;
                    }else{
                        int R = 0;
                        int G = 255;
                        int B = 0;
                        //baryinterp(R, G, B, pointInTriangle, tWorld);
                        float a = computeAmbient(pointInTriangle, l, tWorld.m);
                        float d = computeDiffuse(pointInTriangle, l, tWorld);
                        float s = computeSpecular(pointInTriangle, l, tWorld, e);

                        R = R * a + R * d + R * s;
                        G = G * a + G * d + G * s;
                        B = B * a + B * d + G * s;

                        R = (R < 0) ? 0 : (R > 255) ? 255 : R;
                        G = (G < 0) ? 0 : (G > 255) ? 255 : G;
                        B = (B < 0) ? 0 : (B > 255) ? 255 : B;

                        image[ystep][xstep * 3] = R;
                        image[ystep][xstep * 3 + 1] = G;
                        image[ystep][xstep * 3 + 2] = B;
                    }

                }
            }
        }
    }
}

void outputImage(ofstream &image, vector<vector<int>> &imageBuffer, int width, int height)
{
    image << "P3" << endl;
    image << "#" << endl;
    image << "128 128" << endl;
    image << "255" << endl;
    for (int ystep = height - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < width; xstep++)
        {
            image << imageBuffer[ystep][xstep * 3] << " "
                  << imageBuffer[ystep][xstep * 3 + 1] << " "
                  << imageBuffer[ystep][xstep * 3 + 2] << " ";
        }
        image << endl;
    }
}

int main(int argc, char **argv)
{
    vec3 lightLocation = vec3{0.5, 0.5, 0};
    light l = light{lightLocation, vec3{0.0, 0.0, 1.0}, 0.1, 0.5, 0.5};
    eye e = eye(vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 1, 0), 90.0);
    triangle t(vec3(61, 10, 1), vec3(100, 100, 1), vec3(25, 90, 1), Material{0.9, 0.25, 0.1});
    //question 1a)
    {
        ofstream image("./out/abg.ppm");
        vector<int> row(128 * 3, 129);
        vector<vector<int>> imageBuffer(128, row);
        setupImage(imageBuffer, 128, 128);
        drawImageHalfPlaneTest(imageBuffer, e, t, 128, 128);
        outputImage(image, imageBuffer, 128, 128);
        image.close();
    }

    //question 1b)
    {
        ofstream imageb("./out/colour.ppm");
        vector<int> rowB(128 * 3, 129);
        vector<vector<int>> imageBufferB(128, rowB);
        setupImage(imageBufferB, 128, 128);
        drawImage(imageBufferB, e, t, 128, 128);
        outputImage(imageb, imageBufferB, 128, 128);
        imageb.close();
    }
    //question 1c)
    {
        ofstream imagec("./out/ambient.ppm");
        vector<int> rowC(128 * 3, 129);
        vector<vector<int>> imageBufferC(128, rowC);
        setupImage(imageBufferC, 128, 128);
        drawImageAmbient(imageBufferC, e, t, l, 128, 128);
        outputImage(imagec, imageBufferC, 128, 128);
        imagec.close();
    }

    {
        ofstream imaged("./out/diffuse.ppm");
        vector<int> rowD(128 * 3, 129);
        vector<vector<int>> imageBufferD(128, rowD);
        setupImage(imageBufferD, 128, 128);
        drawImageDiffuse(imageBufferD, e, t, l, 128, 128);
        outputImage(imaged, imageBufferD, 128, 128);
        imaged.close();
    }

    {
        ofstream imagee("./out/specular.ppm");
        vector<int> rowE(128 * 3, 129);
        vector<vector<int>> imageBufferE(128, rowE);
        setupImage(imageBufferE, 128, 128);
        drawImageSpecular(imageBufferE, e, t, l, 128, 128);
        outputImage(imagee, imageBufferE, 128, 128);
        imagee.close();
    }

    {
        ofstream imagee("./out/fullyLit.ppm");
        vector<int> rowE(128 * 3, 129);
        vector<vector<int>> imageBufferE(128, rowE);
        setupImage(imageBufferE, 128, 128);
        drawImageWithLighting(imageBufferE, e, t, l, 128, 128);
        outputImage(imagee, imageBufferE, 128, 128);
        imagee.close();
    }

    {
        //setup ground plane
        vec3 lightLoc = vec3{1, 1, -1};
        light l = light{lightLoc, vec3{0.0, 0.0, -1.0}, 0.1, 0.5, 0.5};
        triangle t1(vec3(0, 0, 1), vec3(128, 0, 1), vec3(0, 0, 3), Material{0.9, 0.25, 0.1});
        triangle t2(vec3(128, 0, 1), vec3(128, 0, 3), vec3(0, 0, 3), Material{0.9, 0.25, 0.1});
        vector<triangle> tris;
        tris.push_back(t1);
        tris.push_back(t2);
        tris.push_back(t);
        ofstream imagee("./out/tes.ppm");
        vector<int> rowE(128 * 3, 129);
        vector<vector<int>> imageBufferE(128, rowE);
        setupImage(imageBufferE, 128, 128);
        //use shadow rays
        drawTriangles(imageBufferE, e, tris, l, 128, 128);
        outputImage(imagee, imageBufferE, 128, 128);
        imagee.close();
    }
}