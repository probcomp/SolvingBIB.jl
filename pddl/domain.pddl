(define (domain bib-gridworld)
    (:requirements :fluents :adl :typing)
    (:types key shape - item lock)
    (:predicates (has ?i - item) (locked ?l - lock))
    (:functions (agentid) (xpos) (ypos) - integer
                (xloc ?o - object) (yloc ?o - object) - integer
                (walls) (barriers) - bit-matrix)
    (:action pickup
     :parameters (?i - item)
     :precondition (and (not (has ?i))
                        (or (= xpos (xloc ?i))
                            (= (- xpos 1) (xloc ?i)) (= (+ xpos 1) (xloc ?i)))
                        (or (= ypos (yloc ?i))
                            (= (- ypos 1) (yloc ?i)) (= (+ ypos 1) (yloc ?i)))
                        (or (and (= (get-index barriers ypos (xloc ?i)) false)
                                 (= (get-index walls ypos (xloc ?i)) false))
                            (and (= (get-index barriers (yloc ?i) xpos) false)
                                 (= (get-index walls (yloc ?i) xpos) false))))
     :effect (and (has ?i) (assign (xloc ?i) (xpos)) (assign (yloc ?i) (ypos)))
    )
    (:action unlock
     :parameters (?k - key ?l - lock)
     :precondition (and (has ?k) (locked ?l)
                        (or (= xpos (xloc ?l))
                            (= (- xpos 1) (xloc ?l)) (= (+ xpos 1) (xloc ?l)))
                        (or (= ypos (yloc ?l))
                            (= (- ypos 1) (yloc ?l)) (= (+ ypos 1) (yloc ?l)))
                        (or (and (= (get-index barriers ypos (xloc ?l)) false)
                                 (= (get-index walls ypos (xloc ?l)) false))
                            (and (= (get-index barriers (yloc ?l) xpos) false)
                                 (= (get-index walls (yloc ?l) xpos) false))))
     :effect (and (not (has ?k)) (forall (?ll - lock) (not (locked ?ll)))
                  (assign barriers
                          (new-bit-matrix false (height barriers)
                                                (width barriers))))
    )
    (:action north
     :precondition (and (> ypos 1)
                        (= (get-index walls (- ypos 1) xpos) false)
                        (= (get-index barriers (- ypos 1) xpos) false))
     :effect (and (decrease ypos 1)
                  (forall (?i - item) (when (has ?i) (decrease (yloc ?i) 1))))
    )
    (:action northwest
     :precondition (and (> ypos 1) (> xpos 1)
                        (= (get-index barriers (- ypos 1) (- xpos 1)) false)
                        (= (get-index walls (- ypos 1) (- xpos 1)) false)
                        (= (get-index barriers ypos (- xpos 1)) false)
                        (= (get-index walls ypos (- xpos 1)) false)
                        (= (get-index barriers (- ypos 1) xpos) false)
                        (= (get-index walls (- ypos 1) xpos) false))
     :effect (and (decrease ypos 1) (decrease xpos 1)
                  (forall (?i - item)
                          (when (has ?i) (and (decrease (yloc ?i) 1)
                                              (decrease (xloc ?i) 1)))))
    )
    (:action west
     :precondition (and (> xpos 1)
                        (= (get-index walls ypos (- xpos 1)) false)
                        (= (get-index barriers ypos (- xpos 1)) false))
     :effect (and (decrease xpos 1)
                  (forall (?i - item) (when (has ?i) (decrease (xloc ?i) 1))))
    )
    (:action southwest
     :precondition (and (< ypos (height walls)) (> xpos 1)
                        (= (get-index barriers (+ ypos 1) (- xpos 1)) false)
                        (= (get-index walls (+ ypos 1) (- xpos 1)) false)
                        (= (get-index barriers ypos (- xpos 1)) false)
                        (= (get-index walls ypos (- xpos 1)) false)
                        (= (get-index barriers (+ ypos 1) xpos) false)
                        (= (get-index walls (+ ypos 1) xpos) false))
     :effect (and (increase ypos 1) (decrease xpos 1)
                  (forall (?i - item)
                          (when (has ?i) (and (increase (yloc ?i) 1)
                                              (decrease (xloc ?i) 1)))))
    )
    (:action south
     :precondition (and (< ypos (height walls))
                        (= (get-index walls (+ ypos 1) xpos) false)
                        (= (get-index barriers (+ ypos 1) xpos) false))
     :effect (and (increase ypos 1)
                  (forall (?i - item) (when (has ?i) (increase (yloc ?i) 1))))
    )
    (:action southeast
     :precondition (and (< ypos (height walls)) (< xpos (width walls))
                        (= (get-index barriers (+ ypos 1) (+ xpos 1)) false)
                        (= (get-index walls (+ ypos 1) (+ xpos 1)) false)
                        (= (get-index barriers ypos (+ xpos 1)) false)
                        (= (get-index walls ypos (+ xpos 1)) false)
                        (= (get-index barriers (+ ypos 1) xpos) false)
                        (= (get-index walls (+ ypos 1) xpos) false))
     :effect (and (increase ypos 1) (increase xpos 1)
                  (forall (?i - item)
                          (when (has ?i) (and (increase (yloc ?i) 1)
                                              (increase (xloc ?i) 1)))))
    )
    (:action east
     :precondition (and (< xpos (width walls))
                        (= (get-index walls ypos (+ xpos 1)) false)
                        (= (get-index barriers ypos (+ xpos 1)) false))
     :effect (and (increase xpos 1)
                  (forall (?i - item) (when (has ?i) (increase (xloc ?i) 1))))
    )
    (:action northeast
     :precondition (and (> ypos 1) (< xpos (width walls))
                        (= (get-index barriers (- ypos 1) (+ xpos 1)) false)
                        (= (get-index walls (- ypos 1) (+ xpos 1)) false)
                        (= (get-index barriers ypos (+ xpos 1)) false)
                        (= (get-index walls ypos (+ xpos 1)) false)
                        (= (get-index barriers (- ypos 1) xpos) false)
                        (= (get-index walls (- ypos 1) xpos) false))
     :effect (and (decrease ypos 1) (increase xpos 1)
                  (forall (?i - item)
                          (when (has ?i) (and (decrease (yloc ?i) 1)
                                              (increase (xloc ?i) 1)))))
    )
)
