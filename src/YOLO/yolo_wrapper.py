from ultralytics import YOLO, RTDETR
import cv2
import numpy as np
from collections import Counter, deque
import supervision as sv
import os
import torch
from torchvision.ops import box_iou
from P2PNet.P2PNet_wrapper import P2PNetWrapper
from P2PNet.P2PNet_config import P2PNetConfig
from sklearn.cluster import DBSCAN

class YoloWrapper:
    def __init__(self, config):
        self.config = config

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.device)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.model = YOLO(config.model_path)
        # self.model = RTDETR(config.model_path)
        self.model.to(0)

        self.prev_state = {} #Para cada trackid (box detectada) guardo sus ultimos dos estados en una queue
        self.tracker = sv.ByteTrack(config.track_thresh, config.match_thresh, config.track_buffer)

        self.crowd_estimator = P2PNetWrapper( P2PNetConfig() )

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
                frame_bgr = self._plot_yolo_boxes(frame_bgr, merged_xyxy, merged_cls, merged_conf, merged_amount, [], res.names)
                self.prev_state.clear()

            else:
                ids   = track.tracker_id.astype(int)
                boxes = track.xyxy.astype(np.float32)
                classes = track.class_id.astype(int)
                confs   = track.confidence.astype(np.float32)
                tracked_amount = np.zeros(len(ids), dtype=np.float32)
                points_list = []
                absolute_points = []

                clamped = boxes.copy()

                pedestrian_mask = np.array([res.names[int(c)] == "pedestrian" for c in classes], dtype=bool)

                #para cada tracked id le hago un clamp transition, esto regula cuanto se puede agrandar y cuanto se puede mover la bounding box haciendo clip
                for i, tid in enumerate(ids):
                    if not pedestrian_mask[i]:
                        continue #solo promedio las merged boxes de pedestrians
                    history = self.prev_state.setdefault(tid, {"boxes_queue": deque(maxlen=20), "boxes_count_queue": deque(maxlen=40)})
                    history["boxes_queue"].append(boxes[i])
                    clamped[i] = self._clamp_transition(history["boxes_queue"], boxes[i])
                    _ , cnt, pts_abs = self.crowd_estimator.infer_on_bbox(frame_bgr, boxes[i])
                    history["boxes_count_queue"].append(cnt)
                    tracked_amount[i] = np.mean(history["boxes_count_queue"])
                    if pts_abs:
                        points_list.extend(pts_abs)

                    absolute_points = (
                        np.asarray(points_list, dtype=np.int32) if points_list else np.empty((0, 2), np.int32)
                    )


                active_ids = set(ids.tolist())
                self.prev_state = {tid: hist for tid, hist in self.prev_state.items() if tid in active_ids}

                frame_bgr = self._plot_yolo_boxes(
                        frame_bgr, clamped, classes, confs, tracked_amount, absolute_points, res.names
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
    
    def _plot_yolo_boxes(self, frame_bgr, xyxy, cls, conf, merge_box_amount, absolute_points, class_names):
        return self._draw_boxes_with_custom_legend(frame_bgr, xyxy, cls, conf, class_names, absolute_points, merge_box_amount)
    
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

            expand_pad = torch.tensor([-100, -100, 100, 100], dtype=torch.float32)
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
                merged_box = self._merge_group_boxes(group_boxes)
                merged_boxes.append(merged_box.tolist())
                merged_classes.append(int(class_id))
                merged_confs.append(float(group_conf.max()))
                merge_box_amount.append(len(comp))

        return (
            np.asarray(merged_boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(merged_classes, dtype=np.int32),
            np.asarray(merged_confs, dtype=np.float32),
            np.asarray(merge_box_amount, dtype=np.int32)
        )

    def _merge_group_boxes(self, group_boxes, keep_frac=0.6):
        #Me quedo con el 60% de las boxes mas cercanas a la mediana de centroides, reduzco el jitter.
        boxes = np.asarray(group_boxes, dtype=np.float32)
        n = boxes.shape[0]

        cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
        cy = 0.5 * (boxes[:, 1] + boxes[:, 3])

        med_cx, med_cy = np.mean(cx), np.mean(cy)

        d = np.abs(cx - med_cx) + np.abs(cy - med_cy) #uso distancia l1
        k = int(np.ceil(keep_frac * n))
        keep_idx = np.argsort(d)[:k]
        core = boxes[keep_idx]

        x1 = float(np.min(core[:, 0]))
        y1 = float(np.min(core[:, 1]))
        x2 = float(np.max(core[:, 2]))
        y2 = float(np.max(core[:, 3]))
        trimmed = np.array([x1, y1, x2, y2], dtype=np.float32)

        return trimmed

    def _color_for_class(self, c):
        rng = np.random.RandomState(c * 9973 + 12345)
        return tuple(int(x) for x in rng.randint(0, 256, size=3))

    def _draw_boxes_with_custom_legend(
            self,
            img_bgr,
            boxes, classes, confs,
            class_names,
            absolute_points,
            merge_box_amount,
        ):

        out = img_bgr

        # proceso las bounding boxes con mayor merge amount al final asi quedan superpuestas a las demas
        order = np.argsort(merge_box_amount, kind="stable")

        for i in order:
            x1, y1, x2, y2 = boxes[i]
            c = int(classes[i])
            cf = confs[i]
            merge_amount = merge_box_amount[i]
            c = int(c)
            col = self._color_for_class(c)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            if class_names[c] == "pedestrian" and merge_amount > 1:
                label = f"Quiet crowd of {int(merge_amount)}"
            else:
                label = f"{class_names[c]}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), col, -1)
            cv2.putText(out, label, (int(x1) + 2, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        for x, y in absolute_points:
            cv2.circle(out, (int(x), int(y)), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        return out


