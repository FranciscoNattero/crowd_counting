from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter, deque
import supervision as sv
import os
import torch
from torchvision.ops import box_iou

class YoloWrapper:
    def __init__(self, config):
        self.config = config

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.device)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.model = YOLO(config.model_path)
        self.model.to(0)

        self.prev_state = {} #Para cada trackid (box detectada) guardo sus ultimos dos estados en una queue
        self.tracker = sv.ByteTrack(config.track_thresh, config.match_thresh, config.track_buffer)

    def predict_image(self, src_path, out_path="annotated_image.jpg",):
        conf, iou = self.config.conf, self.config.iou
        results = self.model.predict(source=src_path, conf=conf, iou=iou, classes=self.config.classes)
        res = results[0]
        count = self._count_people(res)

        img_bgr = self._plot_yolo_boxes(res)
        # self._draw_overlay(img_bgr, f"Cantidad de Personas: {count}")

        cv2.imwrite(out_path, img_bgr)
        return out_path, count

    def predict_video(self, src_path, out_path="annotated_video.mp4", display=False):
        conf, iou = self.config.conf, self.config.iou

        cap = cv2.VideoCapture(str(src_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        writer = None
        total_frames = 0
        self.model.model.float()
        stream = self.model.predict(
                source=str(src_path),
                classes=self.config.classes,
                conf=conf,
                iou=iou,
                stream=True,
                device=self.config.device,
                workers=self.config.workers,
                vid_stride=self.config.vid_stride,
                imgsz=self.config.imgsz,
                half=torch.cuda.is_available()
            )


        for res in stream:
            frame_bgr = np.ascontiguousarray(res.orig_img.copy())
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(np.int32)
            confs = boxes.conf.cpu().numpy().astype(np.float32)

            merged_xyxy, merged_cls, merged_conf, merged_amount = self.__merge_overlapping_boxes_by_class(
                xyxy, cls, confs, res.names
            )
            
            detections = sv.Detections(xyxy=merged_xyxy, class_id=merged_cls, confidence=merged_conf)
            detections.data["merge_box_amount"] = merged_amount

            track = self.tracker.update_with_detections(detections)

            if track.xyxy.shape[0] == 0:
                frame_bgr = self._plot_yolo_boxes(res, merged_xyxy, merged_cls, merged_conf, merged_amount, res.names)
                self.prev_state.clear()
            
            else:
                ids   = track.tracker_id.astype(int)
                boxes = track.xyxy.astype(np.float32)
                classes = track.class_id.astype(int)
                confs   = track.confidence.astype(np.float32)
                tracked_amount = track.data.get(
                        "merge_box_amount",
                        np.ones(len(boxes), dtype=np.int32)
                    )
                
                clamped = boxes.copy()

                #para cada tracked id le hago un clamp transition, esto regula cuanto se puede agrandar y cuanto se puede mover la bounding box haciendo clip
                for i, tid in enumerate(ids):
                    history = self.prev_state.setdefault(tid, deque(maxlen=10))
                    clamped[i] = self._clamp_transition(history, boxes[i])
                    history.append(boxes[i])

                active_ids = set(ids.tolist())
                self.prev_state = {tid: hist for tid, hist in self.prev_state.items() if tid in active_ids}

                frame_bgr = self._plot_yolo_boxes(
                        frame_bgr, clamped, classes, confs, tracked_amount, res.names
                    )

            if writer is None:
                h, w = frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), True)

            # count = self._count_people(merged_cls, res.names)
            # self._draw_overlay(frame_bgr, f"Cantidad de Personas: {count}")

            if display:
                cv2.imshow("detección en tiempo real", frame_bgr)

            writer.write(frame_bgr)
            total_frames += 1

        if writer is not None:
            writer.release()
        return out_path, total_frames
    
    def _draw_overlay(self, img_bgr, text, alpha=0.5, pad=12):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x0, y0, x1, y1 = pad, pad, pad + tw + 2*pad, pad + th + 2*pad

        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (40, 40, 40), -1)
        cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)

        cv2.putText(img_bgr, text, (x0 + pad, y1 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
    def _count_people(self, classes, class_names):
        # print(names)
        return int(sum(1 for c in classes if class_names.get(int(c), "") in ["pedestrian", "people"]))
    
    def _plot_yolo_boxes(self, frame_bgr, xyxy, cls, conf, merge_box_amount, class_names):
        return self._draw_boxes_with_custom_legend(frame_bgr, xyxy, cls, conf, class_names, merge_box_amount)

    def _boxes_iou(self, a, b):
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
        area_b = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
    
    def _to_cxcywh(self, b):
        x1,y1,x2,y2 = b; w=x2-x1; h=y2-y1
        return np.array([x1+0.5*w, y1+0.5*h, w, h], np.float32) #Te devuelve cx, cy, w, h
    
    def _to_xyxy(self, s):
        cx,cy,w,h = s
        return np.array([cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h], np.float32) #te devuelve x1, y1, x2, y2

    def _clamp_transition(self, past_states, curr_state):
        if not past_states:
            return curr_state

        history = list(past_states)
        history.append(curr_state)

        cxcywh = np.stack([self._to_cxcywh(box) for box in history], axis=0)
        avg = cxcywh.mean(axis=0).astype(np.float32)

        return self._to_xyxy(avg)


    def __merge_overlapping_boxes_by_class(self, xyxy, cls, conf, class_names):
        xyxy = np.asarray(xyxy, dtype=np.float32) # (N, 4), para cada box detectada tiene x1, y1, x2, y2
        cls = np.asarray(cls, dtype=np.int32) # (N,), tiene para cada box detectada el numero de la clase detectada
        conf = np.asarray(conf, dtype=np.float32) # (N, ) confidence score para cada box detectada

        merged_boxes = []
        merged_classes = []
        merged_confs = []
        merge_box_amount = []

        for class_id in np.unique(cls):
            idx = np.where(cls == class_id)[0]
            if idx.size == 0:
                continue

            class_label = class_names.get(int(class_id), "")

            boxes_c = xyxy[idx]
            conf_c = conf[idx]

            if class_label != "pedestrian":
                merged_boxes.extend(boxes_c.tolist())
                merged_classes.extend(cls[idx].tolist())
                merged_confs.extend(conf_c.tolist())
                merge_box_amount.extend([1] * len(idx))
                continue

            n = boxes_c.shape[0]

            if n == 0:
                continue

            expand_pad = torch.tensor([-30, -30, 30, 30], dtype=torch.float32)
            boxes_tensor = torch.from_numpy(boxes_c)
            adjacency = (box_iou(boxes_tensor + expand_pad, boxes_tensor + expand_pad)
                         >= self.config.merge_iou_threshold[0]).numpy()


            #Nodo por nodo recorro sus vecinos y los mergeo con el nodo actual. Hago bfs
            visited = np.zeros(n, dtype=bool)
            for i in range(n):
                if visited[i]:
                    continue
                stack = [i]
                comp = []
                visited[i] = True
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    neighbours = np.where(adjacency[u])[0]
                    for v in neighbours:
                        if not visited[v]:
                            visited[v] = True
                            stack.append(v)

                group_boxes = boxes_c[comp]
                group_conf = conf_c[comp]
                merged_boxes.append([
                    float(group_boxes[:, 0].min()),
                    float(group_boxes[:, 1].min()),
                    float(group_boxes[:, 2].max()),
                    float(group_boxes[:, 3].max())
                ])
                merged_classes.append(int(class_id))
                merged_confs.append(float(group_conf.max()))
                merge_box_amount.append(len(comp))

        return (
            np.asarray(merged_boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(merged_classes, dtype=np.int32),
            np.asarray(merged_confs, dtype=np.float32),
            np.asarray(merge_box_amount, dtype=np.int32)
        )

    def _color_for_class(self, c):
        rng = np.random.RandomState(c * 9973 + 12345)
        return tuple(int(x) for x in rng.randint(0, 256, size=3))  # BGR

    def _draw_boxes_with_custom_legend(
            self,
            img_bgr,
            boxes, classes, confs,
            class_names,
            merge_box_amount,
            legend_order=None,
            thickness=2,
            font_scale=0.5
        ):

        legend_title="Detections"
        out = img_bgr

        for (x1, y1, x2, y2), c, cf, merge_amount in zip(boxes, classes, confs, merge_box_amount):
            c = int(c)
            col = self._color_for_class(c)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, thickness)
            # label = f"{class_names[c]} {cf:.2f}"
            if class_names[c] == "pedestrian":
                label = f"Quiet crowd of {merge_amount}"
            else:
                label = f"{class_names[c]}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(out, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), col, -1)
            cv2.putText(out, label, (int(x1) + 2, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)

        counts = Counter(classes.tolist())

        pad = 8
        line_h = 18
        entries = legend_order if legend_order is not None else sorted(counts.keys())
        legend_lines = []
        for c in entries:
            if c in counts:
                legend_lines.append((c, f"{class_names[c]}: {counts[c]}"))

        if legend_title or legend_lines:
            max_text_w = 0
            for txt in [legend_title] + [t for _, t in legend_lines]:
                (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                max_text_w = max(max_text_w, tw)
            box_w = max_text_w + 3*pad + 16
            box_h = pad + line_h + len(legend_lines)*line_h + pad

            cv2.rectangle(out, (pad, pad), (pad + box_w, pad + box_h), (0,0,0), -1)
            cv2.rectangle(out, (pad, pad), (pad + box_w, pad + box_h), (255,255,255), 1)

            cv2.putText(out, legend_title, (pad + 8, pad + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            y = pad + line_h*2
            for c, txt in legend_lines:
                col = self._color_for_class(int(c))
                cv2.rectangle(out, (pad + 8, y - 12), (pad + 8 + 12, y - 12 + 12), col, -1)
                cv2.putText(out, txt, (pad + 8 + 16 + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                y += line_h

        return out





