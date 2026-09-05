import torch
import torchvision


def postprocess(prediction, num_classes, confidence_threshold, nms_threshold):
    corners = prediction.new(prediction.shape)
    corners[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    corners[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    corners[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    corners[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = corners[:, :, :4]

    output = []
    for image_prediction in prediction:
        class_confidence, class_prediction = torch.max(
            image_prediction[:, 5:5 + num_classes], dim=1, keepdim=True)
        keep = image_prediction[:, 4] * class_confidence.squeeze(1) >= confidence_threshold
        detections = torch.cat(
            (image_prediction[:, :5], class_confidence, class_prediction.float()), dim=1
        )[keep]
        if not detections.numel():
            output.append(None)
            continue
        keep = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_threshold,
        )
        output.append(detections[keep])
    return output
