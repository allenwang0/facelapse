// Apple Vision face landmarks for facelapse (macOS only). Adapted from
// face-lapse tools/detect.m (commit 50de507).
//
// Reads image paths on stdin, one per line, and writes one JSON line per
// image to stdout: {"faces": [...]} with, per face, the bounding box and the
// landmark points in pixels (top-left origin), plus yaw/pitch/roll in
// radians as Vision reports them. Unit conversion and keypoint assembly
// happen in vision.py so they can be tested without a Mac.
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <ImageIO/ImageIO.h>

static NSArray *toPixels(VNFaceLandmarkRegion2D *r, VNFaceObservation *f, double W, double H) {
    if (!r || r.pointCount == 0) return @[];
    NSMutableArray *out = [NSMutableArray arrayWithCapacity:r.pointCount];
    const CGPoint *p = r.normalizedPoints;
    CGRect b = f.boundingBox;
    for (NSUInteger i = 0; i < r.pointCount; i++) {
        double x = (b.origin.x + p[i].x * b.size.width) * W;
        double y = (1.0 - (b.origin.y + p[i].y * b.size.height)) * H;
        [out addObject:@[@(x), @(y)]];
    }
    return out;
}

static NSDictionary *describe(NSString *path) {
    NSURL *url = [NSURL fileURLWithPath:path];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
    if (!src) return @{@"error": @"open"};
    CGImageRef im = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    CFRelease(src);
    if (!im) return @{@"error": @"decode"};
    double W = CGImageGetWidth(im), H = CGImageGetHeight(im);
    VNDetectFaceLandmarksRequest *req = [VNDetectFaceLandmarksRequest new];
    req.revision = VNDetectFaceLandmarksRequestRevision3;
    VNImageRequestHandler *h = [[VNImageRequestHandler alloc] initWithCGImage:im options:@{}];
    NSError *err = nil;
    [h performRequests:@[req] error:&err];
    CGImageRelease(im);
    if (err) return @{@"error": err.localizedDescription ?: @"vision"};
    NSMutableArray *faces = [NSMutableArray array];
    for (VNFaceObservation *f in req.results) {
        VNFaceLandmarks2D *l = f.landmarks;
        CGRect b = f.boundingBox;
        NSMutableDictionary *d = [@{
            @"bbox": @[@(b.origin.x * W), @((1.0 - b.origin.y - b.size.height) * H),
                       @((b.origin.x + b.size.width) * W), @((1.0 - b.origin.y) * H)],
            @"confidence": @(f.confidence),
            @"left_pupil": toPixels(l.leftPupil, f, W, H),
            @"right_pupil": toPixels(l.rightPupil, f, W, H),
            @"left_eye": toPixels(l.leftEye, f, W, H),
            @"right_eye": toPixels(l.rightEye, f, W, H),
            @"nose": toPixels(l.nose, f, W, H),
            @"outer_lips": toPixels(l.outerLips, f, W, H),
        } mutableCopy];
        if (f.yaw) d[@"yaw"] = f.yaw;
        if (f.pitch) d[@"pitch"] = f.pitch;
        if (f.roll) d[@"roll"] = f.roll;
        [faces addObject:d];
    }
    return @{@"faces": faces};
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        char line[8192];
        while (fgets(line, sizeof line, stdin)) {
            @autoreleasepool {
                NSString *path = [[NSString stringWithUTF8String:line]
                    stringByTrimmingCharactersInSet:[NSCharacterSet newlineCharacterSet]];
                NSData *json = [NSJSONSerialization dataWithJSONObject:describe(path) options:0 error:nil];
                fwrite(json.bytes, 1, json.length, stdout);
                fputc('\n', stdout);
                fflush(stdout);
            }
        }
    }
    return 0;
}
