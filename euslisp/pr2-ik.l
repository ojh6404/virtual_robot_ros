(require :pr2-interface "package://pr2eus/pr2-interface.l")
(ros::load-ros-manifest "roseus")
(ros::roseus-add-msgs "eus_imitation_msgs")
(ros::roseus-add-srvs "eus_imitation_msgs")
(ros::roseus-add-msgs "sensor_msgs")
(ros::roseus "eus_ik_server")

(ros::set-logger-level "ros.roseus" 5)
(ros::load-ros-manifest "sensor_msgs")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "std_srvs")

(setq ik-stop-step 5)
(setq rate 30)

(defun pr2-ik (arm target-coords)
    (if (eq arm :arms)
      (send *pr2* :inverse-kinematics (list (car target-coords) (cadr target-coords))
             :rotation-axis (list t t)
             :move-target (list (send *pr2* :larm :end-coords) (send *pr2* :rarm :end-coords))
             :link-list (list (send *pr2* :link-list (send (send *pr2* :larm :end-coords) :parent)
                                                      (car (send *pr2* :larm :links)))
                              (send *pr2* :link-list (send (send *pr2* :rarm :end-coords) :parent)
                                                      (car (send *pr2* :rarm :links))))
             :avoid-collision-distance 200 :revert-if-fail nil :stop ik-stop-step
             :debug-view nil)
      (send *pr2* :inverse-kinematics (car target-coords)
             :rotation-axis t :move-target (send *pr2* :rarm :end-coords)
             :link-list (send *pr2* :link-list (send (send *pr2* :rarm :end-coords) :parent)
                                               (car (send *pr2* :rarm :links)))
             :avoid-collision-distance 200 :revert-if-fail nil :stop ik-stop-step
             :debug-view nil)))

(defun eus-ik-cb (req)
  (let* ((current-joint-states (send req :current_joint_states))
         (current-joint-states-name (send current-joint-states :name))
         (current-joint-states-position (send current-joint-states :position))
         (target-joint-state-msg (instance sensor_msgs::JointState :init))
         (joint-list (send *pr2* :joint-list))
         (joint-names (mapcar #'(lambda (joint) (send joint :name)) joint-list))
         (target-coords (mapcar #'(lambda (coords) ;; convert unit from m to mm
                                         (make-coords :pos (float-vector (* 1000 (send coords :position :x))
                                                                        (* 1000 (send coords :position :y))
                                                                        (* 1000 (send coords :position :z)))
                                                      :rot (quaternion2matrix (float-vector (send coords :orientation :w)
                                                                                            (send coords :orientation :x)
                                                                                            (send coords :orientation :y)
                                                                                            (send coords :orientation :z)))))
                                     (send req :target_coords)))
         (arm (if (eq (length (send req :move_end_coords)) 2) :arms
                  (if (eq (car (send req :move_end_coords)) :rarm) :rarm :larm)))
         (res (instance eus_imitation_msgs::RobotIKResponse :init)))
    ;; apply current joint states before IK
    ;; current-joint-states include all joint even fixed joint, so we need to filter it.
    ;; and we need to convert the unit of joint angle from radian to degree and m to mm.
    (dotimes (i (length current-joint-states-name))
      (let* ((joint-name (elt current-joint-states-name i))
             (joint-angle (elt current-joint-states-position i))
             (joint (position joint-name joint-names :test #'string=)))
        (if joint
            (send (elt joint-list joint) :joint-angle (if (derivedp (elt joint-list joint) linear-joint)
                                                          (* 1000.0 joint-angle)
                                                          (rad2deg joint-angle))))))

    (pr2-ik arm target-coords)
    (send target-joint-state-msg :name (mapcar #'(lambda (joint) (send joint :name)) joint-list))
    (send target-joint-state-msg :position (mapcar #'(lambda (joint)
                                                       (if (derivedp joint linear-joint)
                                                           (/ (send joint :joint-angle) 1000.0)
                                                           (deg2rad (send joint :joint-angle))))
                                                   joint-list))
    (send res :target_joint_states target-joint-state-msg)
    (ros::ros-info "IK Solved")
    (send *irtviewer* :draw-objects)
  res))

(defun main ()
  (pr2-init)
  (objects (list *pr2*))
  (ros::rate rate) ;; need to set every loop, otherwise it will be too fast. I don't know why.
  (ros::advertise-service "/eus_ik_solver" eus_imitation_msgs::RobotIK #'eus-ik-cb)
  (ros::spin))

(main)
