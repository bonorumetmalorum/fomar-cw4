#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>

using namespace std;

/*
    vec3 to help with vector operations
*/
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

    vec3 operator*(float scalar)
    {
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

    vec3 operator+(float other)
    {
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

/*
    eye/camera which encodes the position, direction from which the scene is viewed, and its up and fov 
*/
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

/*
    vec3 redefinition for colour data
*/
typedef vec3 colour;

/*
    helper function to blend two colours together by simply multiplying them
    @param a colour provided between the range 0-255 per channel
    @param b colout provided between the range 0-1 per channel
*/
colour blend(colour a, colour b){
    return colour(a.x * b.x, a.y * b.y,a.z * b.z);
}

/*
    point light which encodes its colour, position, diffuse intensity, specular intensity and ambient intensity as well as the specular coefficient
*/
struct light
{
    colour rgb;
    vec3 position;
    float diffuseIntensity;
    float specularIntensity;
    float specularCoeff;
    float ambientIntensity;
};

/*
    Material which encodes the colour, diffuse intensity, specular intensity and ambient intensity
*/
struct Material
{
    colour rgb;
    float diffuseIntensity;
    float specularIntensity;
    float ambientIntensity;
};

/*
    triangle which enodes its three vertices
*/
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

/*
    ray which encodes the origin and direction
*/
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

/*
    get the normal of a triangle and as a consequence the normal of the plane the triangle is on
    @param t the triangle for which the normal will calculated provided in world coordinates
    @return vec3 normal of the trianlge/plane in world coordinates
*/
vec3 getPlaneNormal(triangle t)
{
    vec3 planeDim1 = t.B - t.A;
    vec3 planeDim2 = t.C - t.A;
    vec3 normal = cross(planeDim1, planeDim2);
    normal.normalise();
    return normal;
}

/*
    check if a point is inside a triangle by checking if the point is on the left of every edge
    @param t the triangle to check if the point is inside provided in world coordinates
    @param point the point to check for provided in world coordintaes
    @param planeNormal the normal of the plane on which the triangle and point are on provided in world coordinates
    @return true if the point is inside the triangle, false otherwise
*/
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


/*
    check if a ray is intersecting a triangle
    @param r the ray to use for checking intersection provided in world coordinates
    @param t the triangle to check for intersection provided in world coordinates
    @param pointOut the point at which the ray intersects the triangle (only set if there is an intersection)
    @return true if there is an intersection and false otherwise
*/
bool isIntersectingTriangle(ray r, triangle t, vec3 &pointOut)
{
    vec3 normal = getPlaneNormal(t);

    float denom = dot(normal, r.direction);

    if (denom == 0)
    {
        return false;
    }

    vec3 numerator = t.A - r.start;
    float d = dot(numerator, normal) / denom;

    if (d < 0)
    {
        return false;
    }

    vec3 p = r.start + (r.direction * d);
    // cout << p.x << " " << p.y << " " << p.z << endl;
    if (isInsideTriangle(t, p, normal))
    {
        pointOut = p;

        return true;
    }
    return false;
}

/*
    calculate the distance of a point from an edge
    @param point the point to calculate the distance for
    @param vertex1 vertex at the end of the edge
    @param vertex2 vertex at the other end of the edge
    @return float the distance of the point from the edge
*/
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

/*
    helper function to set image colour to light yellow (255, 255, 129)
    @param image the image buffer to setup
    @param width the width of the image
    @param height the height of the image
*/
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

/*
    use barycentric interpolation to determine the colour of a point on a triangle
    @param point the point on the triangle in world coordinates
    @param t the triangle in world coordinates
    @return colour final colour of this point in range 0-255 for R, G, B
*/
colour baryinterp(vec3 point, triangle t)
{
    int R = 0;
    int G = 0;
    int B = 0;
    float alpha = distance(point, t.B, t.C) / distance(t.A, t.B, t.C); //distance from xstep,ystep to CB
    float beta = distance(point, t.A, t.C) / distance(t.B, t.A, t.C);  //distance from xstep,ystep to AC
    float gamma = distance(point, t.B, t.A) / distance(t.C, t.B, t.A); //distance from xstep,ystep to BA
    R = 255 * alpha;
    G = 255 * beta;
    B = 255 * gamma;
    vec3 colour(255,255,129);
    colour.x = (R < 0) ? 0 : (R > 255) ? 255 : R;
    colour.y = (G < 0) ? 0 : (G > 255) ? 255 : G;
    colour.z = (B < 0) ? 0 : (B > 255) ? 255 : B;
    return colour;
};

/*
    cast a ray from the origin to the location
    @param origin the point form which to cast provided in world coordinates
    @param the direction in which to cast provided in world coordinates
    @return ray which encodes the start position and computed direction
*/
ray castray(vec3 origin, vec3 location)
{
    vec3 direction = location - origin;
    //direction.normalise();
    return ray(origin, direction);
}

/*
    convert pixel coordinates into world coordinates (this will only work for square images and an fov of 90 degrees)
    @coord the pixel coordinates
    @width the width of the image 
    @height the height of the image
*/
vec3 convertCoordinates(vec3 coord, int width, int height)
{
    float aspectRatio = float(width) / float(height);
    float xr = ((2 * ((coord.x) / width)) - 1) * aspectRatio;
    float yr = ((2 * ((coord.y) / height)) - 1) * aspectRatio;
    return vec3(xr, yr, coord.z);
}

/*
    cast shadow ray to determine if point is in a shadow
    @param point the point from which to cast the shadow ray
    @normal the normal of the plane on whic the point lives
    @t the triangle to check shadow ray intersection against
    @l the light source to which the shadow ray is cast
    @return true if intersecting with triangle false otherwise
*/
bool isInShadow(vec3 point, vec3 normal, triangle t, light l)
{
    ray r = castray(point + 0.01, l.position);
    vec3 pos;
    cout << r.direction.x << " " << r.direction.y << " " << r.direction.z << endl;
    if (isIntersectingTriangle(r, t, pos))
    {
        cout << "yes" << endl;
        return true;
    }
    return false;
}

/*
    draw the scene using the halfplane test
    @param image image buffer to store the pixels final colour values
    @param e eye from which the scene is viewed provided in world coordinates
    @param t triangle to draw provided in world coordinates
    @param xmax image width
    @param ymax image height
*/
void drawImageHalfPlaneTest(vector<vector<int>> &image, eye e, triangle t, float xmax, float ymax)
{
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, t, pointInTriangle))
            {
                image[ystep][xstep * 3] = 0;
                image[ystep][xstep * 3 + 1] = 0;
                image[ystep][xstep * 3 + 2] = 0;
            }
        }
    }
}

/*
    use barycentric interpolation to draw a triangle in the scene
    @param image image bufer to store final pixel colour values
    @param e the eye from which the scene is viewed provided in world coordinates
    @param t triangle to draw provided in world coordinates
    @param xmax image width
    @param ymax image height
*/
void drawImage(vector<vector<int>> &image, eye e, triangle t, float xmax, float ymax)
{
    //triangle tWorld = triangle(convertCoordinates(t.A, 128, 128), convertCoordinates(t.B, 128, 128), convertCoordinates(t.C, 128, 128), t.m);
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, t, pointInTriangle))
            {
                colour c = baryinterp(pointInTriangle, t);
                image[ystep][xstep * 3] = c.x;
                image[ystep][xstep * 3 + 1] = c.y;
                image[ystep][xstep * 3 + 2] = c.z;
            }
        }
    }
}

/*
    compute the diffuse intensity at a point given the light and triangle
    @param point the point at which to compute the diffuse intensity
    @param l the light souce provided in world coordinates
    @param t the triangle provided in world coordinates
    @return colour the final diffuse colour intensity at this point
*/
colour computeDiffuse(vec3 point, light l, triangle t)
{
    vec3 triangleNormal = getPlaneNormal(t);
    vec3 vl = point - l.position;
    float numerator = dot(triangleNormal, vl);
    float denom = triangleNormal.length() * vl.length();
    return l.rgb * l.diffuseIntensity * t.m.diffuseIntensity * (numerator / denom);
}

/*
    compute the sepcular intensity at a point given a light source, triangle and eye
    @param point the point at which to compute specular intensity
    @param l the light source provided in world coordinates
    @param t the triangle on which the point exists
    @param e the eye from which the scene is viewed
    @return colour the final specular colour intensity at this point
*/
colour computeSpecular(vec3 point, light l, triangle t, eye e)
{
    vec3 triangleNormal = getPlaneNormal(t);
    vec3 vl = l.position - point;
    vec3 ve = e.position - point;
    vec3 vb = (vl + ve) / 2;
    float numerator = dot(triangleNormal, vb);
    float denom = triangleNormal.length() * vb.length();

    float angle = numerator / denom;
    return l.rgb * l.specularIntensity * t.m.specularIntensity * powf(angle, l.specularCoeff);
}

/*
    compute the ambient colour for a point given a light source and a material
    @param point the point at which to compute ambient intensity
    @param l the light source, in world coordinates
    @param m the matieral of the surface on which the point lives
    @return the final colour intensity at this point
*/
colour computeAmbient(vec3 point, light l, Material m)
{
    return l.rgb * (l.ambientIntensity * m.ambientIntensity);
}

/*
    draw triangle with only ambient colour
    image buffer to store final pixel colour
    eye from which to cast rays, provided in world coordinates
    triangle to draw provided in pixel coordinates
    light source provided in world coordinates
    image dimensions
*/
void drawImageAmbient(vector<vector<int>> &image, eye e, triangle t, light l, float xmax, float ymax)
{
    //triangle tWorld = triangle(convertCoordinates(t.A, 128, 128), convertCoordinates(t.B, 128, 128), convertCoordinates(t.C, 128, 128), t.m);
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, t, pointInTriangle))
            {
                //baryinterp(R, G, B, pointInTriangle, tWorld);
                colour amt = blend(t.m.rgb, computeAmbient(pointInTriangle, l, t.m));
                image[ystep][xstep * 3] = amt.x;
                image[ystep][xstep * 3 + 1] = amt.y;
                image[ystep][xstep * 3 + 2] = amt.z;
            }
        }
    }
}

/*
    draw triangle with only specular colour
    image buffer to store final pixel colour
    eye from which to cast rays, provided in world coordinates
    triangle to draw provided in world coordinates
    light source provided in world coordinates
    image dimensions
*/
void drawImageSpecular(vector<vector<int>> &image, eye e, triangle t, light l, float xmax, float ymax)
{
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, t, pointInTriangle))
            {
                int R = 0;
                int G = 255;
                int B = 0;
                //baryinterp(R, G, B, pointInTriangle, tWorld);
                colour amt = blend(t.m.rgb, computeSpecular(pointInTriangle, l, t, e));
                image[ystep][xstep * 3] = amt.x;
                image[ystep][xstep * 3 + 1] = amt.y;
                image[ystep][xstep * 3 + 2] = amt.z;
            }
        }
    }
}

/*
    draw triangle with only diffuse colour
    image buffer to store final pixel colour
    eye from which to cast rays, provided in world coordinates
    triangle to draw provided in world coordinates
    light source provided in world coordinates
    image dimensions
*/
void drawImageDiffuse(vector<vector<int>> &image, eye e, triangle t, light l, float xmax, float ymax)
{
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, t, pointInTriangle))
            {
                int R = 0;
                int G = 255;
                int B = 0;
                //baryinterp(R, G, B, pointInTriangle, tWorld);
                colour amt = blend(t.m.rgb, computeDiffuse(pointInTriangle, l, t));
                image[ystep][xstep * 3] = amt.x;
                image[ystep][xstep * 3 + 1] = amt.y;
                image[ystep][xstep * 3 + 2] = amt.z;
            }
        }
    }
}

/*
    draws the triangle with full ambient + diffuse + specular lighting from the blinn phong lighting model
    image buffer to store final pixel colours
    eye provided in world coordinates, to cast rays from
    triangle to draw, in world coordinates
    light provided in world coordinates
    image dimensions
*/
void drawImageWithLighting(vector<vector<int>> &image, eye e, triangle t, light l, float xmax, float ymax)
{
    for (int ystep = ymax - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < xmax; xstep++)
        {
            ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
            vec3 pointInTriangle;
            if (isIntersectingTriangle(r, t, pointInTriangle))
            {
                //baryinterp(R, G, B, pointInTriangle, tWorld);
                colour a = blend(t.m.rgb, computeAmbient(pointInTriangle, l, t.m));
                colour d = blend(t.m.rgb, computeDiffuse(pointInTriangle, l, t));
                colour s = blend(t.m.rgb, computeSpecular(pointInTriangle, l, t, e));

                colour p = a + d + s;

                p.x = (p.x < 0) ? 0 : (p.x > 255) ? 255 : p.x;
                p.y = (p.y < 0) ? 0 : (p.y > 255) ? 255 : p.y;
                p.z = (p.z < 0) ? 0 : (p.z > 255) ? 255 : p.z;

                image[ystep][xstep * 3] = p.x;
                image[ystep][xstep * 3 + 1] = p.y;
                image[ystep][xstep * 3 + 2] = p.z;
            }
        }
    }
}

/*
    image buffer to output final pixel colours
    draw an array of triangles, sorted from farthest back to closest forward
    draws with blinn-phong lighting
    draws with shadow rays
    triangles must be supplied in world coordinates
    light must be provided in world coordinates
    eye must be provided in world coordinates
    image dimensions must be provided
*/
void drawTriangles(vector<vector<int>> &image, eye e, vector<triangle> tris, light l, float xmax, float ymax)
{
    for (int i = 0; i < tris.size(); i++)
    {
        triangle t = tris[i];
        for (int ystep = ymax - 1; ystep >= 0; ystep--)
        {
            for (int xstep = 0; xstep < xmax; xstep++)
            {
                ray r = castray(e.position, convertCoordinates(vec3(xstep, ystep, 1.0), 128, 128));
                vec3 pointInTriangle;
                vec3 normal = getPlaneNormal(t);
                bool inShadow = false;
                if (isIntersectingTriangle(r, t, pointInTriangle))
                {
                    for (int j = 0; j < tris.size(); j++)
                    {
                        if (j == i)
                        {
                            continue;
                        }
                        if (inShadow)
                        {
                            break;
                        }
                        inShadow = isInShadow(pointInTriangle, normal, tris[j], l);
                    }
                    if (inShadow)
                    {
                        colour a = blend(tris[i].m.rgb, computeAmbient(pointInTriangle, l, t.m));
                        image[ystep][xstep * 3] = a.x;
                        image[ystep][xstep * 3 + 1] = a.y;
                        image[ystep][xstep * 3 + 2] = a.z;
                    }
                    else
                    {
                        int R = 0;
                        int G = 255;
                        int B = 0;
                        //baryinterp(R, G, B, pointInTriangle, tWorld);
                        colour a = blend(tris[i].m.rgb, computeAmbient(pointInTriangle, l, t.m));
                        colour d = blend(tris[i].m.rgb, computeDiffuse(pointInTriangle, l, t));
                        colour s = blend(tris[i].m.rgb, computeSpecular(pointInTriangle, l, t, e));

                        colour p = a + d + s;

                        p.x = (p.x < 0) ? 0 : (p.x > 255) ? 255 : p.x;
                        p.y = (p.y < 0) ? 0 : (p.y > 255) ? 255 : p.y;
                        p.z = (p.z < 0) ? 0 : (p.z > 255) ? 255 : p.z;

                        image[ystep][xstep * 3] = p.x;
                        image[ystep][xstep * 3 + 1] = p.y;
                        image[ystep][xstep * 3 + 2] = p.z;
                    }
                }
            }
        }
    }
}

/*
    write image buffer to ppm file
*/
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
    vec3 lightLocation = vec3{0, 0, -1};
    light l = light{colour(0, 255, 129), lightLocation, 0.01, 0.1, 100.0, 0.1};
    eye e = eye(vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 1, 0), 90.0);
    triangle t(vec3(-0.04688, -0.84375, 1), vec3(0.5625, 0.5625, 1), vec3(-0.60938, 0.40625, 1), Material{colour(125, 125, 125), 0.1, 0.1, 0.01});
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
    //question 1d)
    {
        //setup ground plane
        vec3 lightLoc = vec3{0.5, 0.5, -1};
        light l = light{vec3(11,11,11), lightLoc, 0.1, 0.1, 100.0, 0.1};
        triangle t1(vec3(1, -1, 1), vec3(1, -1, 2), vec3(-1, -1, 2), Material{vec3(255,10,90), 0.1, 0.1, 0.1});
        triangle t2(vec3(-1, -1, 2), vec3(-1, -1, 1), vec3(1, -1, 1), Material{vec3(255,10,90), 0.1, 0.1, 0.1});
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