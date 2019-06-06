#include "partitioning.h"

static int pcount = 1;

#include "../../thirdparty/kfusion.h"

#undef isnan
#undef isfinite

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <execinfo.h>

/*
#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <TooN/GR_SVD.h>
*/
#include <../../include/constant_parameters.h>

#define INVALID -2   // this is used to mark invalid entries in normal or vertex maps

using namespace std;





__global__ void renderDepthKernel(__P_KARGS, Image<uchar4> out, const Image<float> depth,
		const float nearPlane, const float farPlane) {
	//const float d = (clamp(depth.el(), nearPlane, farPlane) - nearPlane) / (farPlane - nearPlane);
	//out.el() = make_uchar3(d * 255, d * 255, d * 255);
  __P_BEGIN; __P_LOOPXY
    
	if (depth.el(__P_HDARGS) < nearPlane)
		out.el(__P_HDARGS) = make_uchar4(255, 255, 255, 0); // The forth value is padding for memory alignement and so it is for following uchar4
	else {
		if (depth.el(__P_HDARGS) > farPlane)
			out.el(__P_HDARGS) = make_uchar4(0, 0, 0, 0); 
		else {
			float h = (depth.el(__P_HDARGS) - nearPlane) / (farPlane - nearPlane);
			h *= 6.0;
			const int sextant = (int) h;
			const float fract = h - sextant;
			const float mid1 = 0.25 + (0.5 * fract);
			const float mid2 = 0.75 - (0.5 * fract);
			switch (sextant) {
			    case 0: out.el(__P_HDARGS) = make_uchar4(191, 255 * mid1, 64, 0); break;
			    case 1: out.el(__P_HDARGS) = make_uchar4(255 * mid2, 191, 64, 0); break;
			    case 2: out.el(__P_HDARGS) = make_uchar4(64, 191, 255 * mid1, 0); break;
			    case 3: out.el(__P_HDARGS) = make_uchar4(64, 255 * mid2, 191, 0); break;
			    case 4: out.el(__P_HDARGS) = make_uchar4(255 * mid1, 64, 191, 0); break;
			    case 5: out.el(__P_HDARGS) = make_uchar4(191, 64, 255 * mid2, 0); break;
			}
			// out.el() = gs2rgb(d);
		}
	}
  __P_LOOPEND
}

__global__ void renderTrackKernel(__P_KARGS, Image<uchar4> out,
		const Image<TrackData> data) {
  __P_BEGIN; __P_LOOPXY
	const uint2 pos = thr2pos2(__P_HDARGS);
	// The forth value is padding for memory alignement and so it is for following uchar4
	switch (data[pos].result) {
	    case  1: out[pos] = make_uchar4(128, 128, 128, 0);	break; // ok
	    case -1: out[pos] = make_uchar4(0, 0, 0, 0);	break; // no input 
	    case -2: out[pos] = make_uchar4(255, 0, 0, 0);	break; // not in image 
	    case -3: out[pos] = make_uchar4(0, 255, 0, 0);	break; // no correspondence
	    case -4: out[pos] = make_uchar4(0, 0, 255, 0);	break; // too far away
	    case -5: out[pos] = make_uchar4(255, 255, 0, 0);	break; // wrong normal
	}
	__P_LOOPEND
}

__global__ void renderVolumeKernel(__P_KARGS, Image<uchar4> render, const Volume volume,
		const Matrix4 view, const float nearPlane, const float farPlane,
		const float step, const float largestep, const float3 light,
		const float3 ambient) {
  __P_BEGIN; __P_LOOPXY
	const uint2 pos = thr2pos2(__P_HDARGS);

	float4 hit = raycast(volume, pos, view, nearPlane, farPlane, step,
			largestep);
	if (hit.w > 0) {
		const float3 test = make_float3(hit);
		const float3 surfNorm = volume.grad(test);
		if (length(surfNorm) > 0) {
			const float3 diff = normalize(light - test);
			const float dir = fmaxf(dot(normalize(surfNorm), diff), 0.f);
			const float3 col = clamp(make_float3(dir) + ambient, 0.f, 1.f)
					* 255;
			render.el(__P_HDARGS) = make_uchar4(col.x, col.y, col.z, 0); // The forth value is padding for memory alignement and so it is for following uchar4
		} else {
			render.el(__P_HDARGS) = make_uchar4(0, 0, 0, 0);
		}
	} else {
		render.el(__P_HDARGS) = make_uchar4(0, 0, 0, 0);
	}
	__P_LOOPEND
}
/*
 void renderVolumeLight( Image<uchar3> out, const Volume & volume, const Matrix4 view, const float nearPlane, const float farPlane, const float largestep, const float3 light, const float3 ambient ){
 dim3 block(16,16);
 raycastLight<<<divup(out.size, block), block>>>( out,  volume, view, nearPlane, farPlane, volume.dim.x/volume.size.x, largestep, light, ambient );
 }
 */

/*
 void renderInput( Image<float3> pos3D, Image<float3> normal, Image<float> depth, const Volume volume, const Matrix4 view, const float nearPlane, const float farPlane, const float step, const float largestep){
 dim3 block(16,16);
 raycastInput<<<divup(pos3D.size, block), block>>>(pos3D, normal, depth, volume, view, nearPlane, farPlane, step, largestep);
 }
 */

__global__ void initVolumeKernel(__P_KARGS, Volume volume, const float2 val) {
  __P_BEGIN; __P_LOOPXY
	uint3 pos = make_uint3(thr2pos2(__P_HDARGS));
	for (pos.z = 0; pos.z < volume.size.z; ++pos.z)
		volume.set(pos, val);
	__P_LOOPEND
}

__global__ void raycastKernel(__P_KARGS, Image<float3> pos3D, Image<float3> normal,
		const Volume volume, const Matrix4 view, const float nearPlane,
		const float farPlane, const float step, const float largestep) {
  __P_BEGIN; __P_LOOPXY
	const uint2 pos = thr2pos2(__P_HDARGS);

	const float4 hit = raycast(volume, pos, view, nearPlane, farPlane, step,
			largestep);
	if (hit.w > 0) {
		pos3D[pos] = make_float3(hit);
		float3 surfNorm = volume.grad(make_float3(hit));
		if (length(surfNorm) == 0) {
			normal[pos].x = INVALID;
		} else {
			normal[pos] = normalize(surfNorm);
		}
	} else {
		pos3D[pos] = make_float3(0);
		normal[pos] = make_float3(INVALID, 0, 0);
	}
	__P_LOOPEND
}

__forceinline__ __device__ float sq(const float x) {
	return x * x;
}

__global__ void integrateKernel(__P_KARGS, Volume vol, const Image<float> depth,
		const Matrix4 invTrack, const Matrix4 K, const float mu,
		const float maxweight) {
  __P_BEGIN; __P_LOOPXY
	uint3 pix = make_uint3(thr2pos2(__P_HDARGS));
	float3 pos = invTrack * vol.pos(pix);
	float3 cameraX = K * pos;
	const float3 delta = rotate(invTrack,
			make_float3(0, 0, vol.dim.z / vol.size.z));
	const float3 cameraDelta = rotate(K, delta);

	for (pix.z = 0; pix.z < vol.size.z; ++pix.z, pos += delta, cameraX +=
			cameraDelta) {
		if (pos.z < 0.0001f) // some near plane constraint
			continue;
		const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f,
				cameraX.y / cameraX.z + 0.5f);
		if (pixel.x < 0 || pixel.x > depth.size.x - 1 || pixel.y < 0
				|| pixel.y > depth.size.y - 1)
			continue;
		const uint2 px = make_uint2(pixel.x, pixel.y);
		if (depth[px] == 0)
			continue;
		const float diff = (depth[px] - cameraX.z)
				* sqrt(1 + sq(pos.x / pos.z) + sq(pos.y / pos.z));
		if (diff > -mu) {
			const float sdf = fminf(1.f, diff / mu);
			float2 data = vol[pix];
			data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.f, 1.f);
			data.y = fminf(data.y + 1, maxweight);
			vol.set(pix, data);
		}
	}
	__P_LOOPEND
}

__global__ void depth2vertexKernel(__P_KARGS, Image<float3> vertex,
		const Image<float> depth, const Matrix4 invK) {
  __P_BEGIN; __P_LOOPXY
	const uint2 pixel = thr2pos2(__P_HDARGS);
	if (pixel.x >= depth.size.x || pixel.y >= depth.size.y)
	  continue;

	if (depth[pixel] > 0) {
		vertex[pixel] = depth[pixel]
				* (rotate(invK, make_float3(pixel.x, pixel.y, 1.f)));
	} else {
		vertex[pixel] = make_float3(0);
	}
	__P_LOOPEND
}

__global__ void vertex2normalKernel(__P_KARGS, Image<float3> normal,
		const Image<float3> vertex) {
  __P_BEGIN; __P_LOOPXY
	const uint2 pixel = thr2pos2(__P_HDARGS);
	if (pixel.x >= vertex.size.x || pixel.y >= vertex.size.y)
	  continue;

	const float3 left = vertex[make_uint2(max(int(pixel.x) - 1, 0), pixel.y)];
	const float3 right = vertex[make_uint2(min(pixel.x + 1, vertex.size.x - 1),
			pixel.y)];
	const float3 up = vertex[make_uint2(pixel.x, max(int(pixel.y) - 1, 0))];
	const float3 down = vertex[make_uint2(pixel.x,
			min(pixel.y + 1, vertex.size.y - 1))];

	if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
		normal[pixel].x = INVALID;
		continue;
	}

	const float3 dxv = right - left;
	const float3 dyv = down - up;
	normal[pixel] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
	__P_LOOPEND
}

template <int HALFSAMPLE>
__global__ void mm2metersKernel(__P_KARGS, Image<float> depth, const Image<ushort> in ) {
  __P_BEGIN; __P_LOOPXY
	const uint2 pixel = thr2pos2(__P_HDARGS);
	depth[pixel] = in[pixel * (HALFSAMPLE+1)] / 1000.0f;
	__P_LOOPEND
}

//column pass using coalesced global memory reads
__global__ void bilateralFilterKernel(__P_KARGS, Image<float> out, const Image<float> in,
		const Image<float> gaussian, const float e_d, const int r) {
  __P_BEGIN; __P_LOOPXY
	       const uint2 pos = thr2pos2(__P_HDARGS);
	       //const uint2 pos= make_uint2(0,0);
	if (in[pos] == 0) {
		out[pos] = 0;
		continue;
	}

	float sum = 0.0f;
	float t = 0.0f;
	const float center = in[pos];

	for (int i = -r; i <= r; ++i) {
		for (int j = -r; j <= r; ++j) {
			const float curPix = in[make_uint2(
					clamp(pos.x + i, 0u, in.size.x - 1),
					clamp(pos.y + j, 0u, in.size.y - 1))];
			if (curPix > 0) {
				const float mod = sq(curPix - center);
				const float factor = gaussian[make_uint2(i + r, 0)]
						* gaussian[make_uint2(j + r, 0)]
						* __expf(-mod / (2 * e_d * e_d));
				t += factor * curPix;
				sum += factor;
			}
		}
	}
	out[pos] = t / sum;
	__P_LOOPEND
}

// filter and halfsample
__global__ void halfSampleRobustImageKernel(__P_KARGS, Image<float> out,
		const Image<float> in, const float e_d, const int r) {
  __P_BEGIN; __P_LOOPXY
	const uint2 pixel = thr2pos2(__P_HDARGS);
	const uint2 centerPixel = 2 * pixel;

	if (pixel.x >= out.size.x || pixel.y >= out.size.y)
	  continue;

	float sum = 0.0f;
	float t = 0.0f;
	const float center = in[centerPixel];
	for (int i = -r + 1; i <= r; ++i) {
		for (int j = -r + 1; j <= r; ++j) {
			float current = in[make_uint2(
					clamp(make_int2(centerPixel.x + j, centerPixel.y + i),
							make_int2(0),
							make_int2(in.size.x - 1, in.size.y - 1)))]; // TODO simplify this!
			if (fabsf(current - center) < e_d) {
				sum += 1.0f;
				t += current;
			}
		}
	}
	out[pixel] = t / sum;
	__P_LOOPEND
}

__global__ void generate_gaussian(__P_KARGS, Image<float> out, float delta, int radius) {
  __P_BEGIN;
  __P_LOOPX
	int x = threadIdx.x - radius;
	out[make_uint2(threadIdx.x, 0)] = __expf(-(x * x) / (2 * delta * delta));
	__P_LOOPEND
}

__global__ void trackKernel(__P_KARGS, Image<TrackData> output,
		const Image<float3> inVertex, const Image<float3> inNormal,
		const Image<float3> refVertex, const Image<float3> refNormal,
		const Matrix4 Ttrack, const Matrix4 view, const float dist_threshold,
		const float normal_threshold) {
  __P_BEGIN; __P_LOOPXY
	const uint2 pixel = thr2pos2(__P_HDARGS);
	if (pixel.x >= inVertex.size.x || pixel.y >= inVertex.size.y)
	  continue;

	TrackData & row = output[pixel];

	if (inNormal[pixel].x == INVALID) {
		row.result = -1;
		continue;
	}

	const float3 projectedVertex = Ttrack * inVertex[pixel];
	const float3 projectedPos = view * projectedVertex;
	const float2 projPixel = make_float2(projectedPos.x / projectedPos.z + 0.5f,
			projectedPos.y / projectedPos.z + 0.5f);

	if (projPixel.x < 0 || projPixel.x > refVertex.size.x - 1 || projPixel.y < 0
			|| projPixel.y > refVertex.size.y - 1) {
		row.result = -2;
		continue;
	}

	const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
	const float3 referenceNormal = refNormal[refPixel];

	if (referenceNormal.x == INVALID) {
		row.result = -3;
		continue;
	}

	const float3 diff = refVertex[refPixel] - projectedVertex;
	const float3 projectedNormal = rotate(Ttrack, inNormal[pixel]);

	if (length(diff) > dist_threshold) {
		row.result = -4;
		continue;
	}
	if (dot(projectedNormal, referenceNormal) < normal_threshold) {
		row.result = -5;
		continue;
	}

	row.result = 1;
	row.error = dot(referenceNormal, diff);
	((float3 *) row.J)[0] = referenceNormal;
	((float3 *) row.J)[1] = cross(projectedVertex, referenceNormal);
	__P_LOOPEND
}

__global__ void reduceKernel(__P_KARGS, float * out, const Image<TrackData> J,
		const uint2 size) {
  __P_BEGIN; __P_LOOPXY
	__shared__
	float S[112][32]; // this is for the final accumulation
	const uint sline = threadIdx.x;

	float sums[32];
	float * jtj = sums + 7;
	float * info = sums + 28;

	for (uint i = 0; i < 32; ++i)
		sums[i] = 0;

	for (uint y = blockid.x; y < size.y; y += blocks.x) {
		for (uint x = sline; x < size.x; x += blockDim.x) {
			const TrackData & row = J[make_uint2(x, y)];
			if (row.result < 1) {
				info[1] += row.result == -4 ? 1 : 0;
				info[2] += row.result == -5 ? 1 : 0;
				info[3] += row.result > -4 ? 1 : 0;
				continue;
			}

			// Error part
			sums[0] += row.error * row.error;

			// JTe part
			for (int i = 0; i < 6; ++i)
				sums[i + 1] += row.error * row.J[i];

			// JTJ part, unfortunatly the double loop is not unrolled well...
			jtj[0] += row.J[0] * row.J[0];
			jtj[1] += row.J[0] * row.J[1];
			jtj[2] += row.J[0] * row.J[2];
			jtj[3] += row.J[0] * row.J[3];
			jtj[4] += row.J[0] * row.J[4];
			jtj[5] += row.J[0] * row.J[5];

			jtj[6] += row.J[1] * row.J[1];
			jtj[7] += row.J[1] * row.J[2];
			jtj[8] += row.J[1] * row.J[3];
			jtj[9] += row.J[1] * row.J[4];
			jtj[10] += row.J[1] * row.J[5];

			jtj[11] += row.J[2] * row.J[2];
			jtj[12] += row.J[2] * row.J[3];
			jtj[13] += row.J[2] * row.J[4];
			jtj[14] += row.J[2] * row.J[5];

			jtj[15] += row.J[3] * row.J[3];
			jtj[16] += row.J[3] * row.J[4];
			jtj[17] += row.J[3] * row.J[5];

			jtj[18] += row.J[4] * row.J[4];
			jtj[19] += row.J[4] * row.J[5];

			jtj[20] += row.J[5] * row.J[5];

			// extra info here
			info[0] += 1;
		}
	}

	for (int i = 0; i < 32; ++i) // copy over to shared memory
		S[sline][i] = sums[i];

	__syncthreads();            // wait for everyone to finish

	if (sline < 32) { // sum up columns and copy to global memory in the final 32 threads
		for (unsigned i = 1; i < blockDim.x; ++i)
			S[0][sline] += S[i][sline];
		out[sline + blockid.x * 32] = S[0][sline];
	}
	__P_LOOPEND_SAFE
}