class Object:
    """
    Class that represent each detected object with its properties.
    Attributes:
        label (str): The label of the object.
        centroid (tuple): The centroid coordinates of the object.
        bbox (tuple, optional): The bounding box of the object. Defaults to None.
        description (str, optional): A textual description of the object. Defaults to "".
        color (str, optional): The color of the object. Defaults to "".
        material (str, optional): The material of the object. Defaults to "".
        shape (str, optional): The shape of the object. Defaults to "".
    """

    def __init__(self, label, centroid, bbox=None, description="", color="",material="",shape="", object_id=None):
        # `label` comes from perception and may change between frames (for
        # example ``stove#2``).  Persistent objects receive an immutable ID in
        # ObjectServices when they are first added to the world model.
        self.object_id = object_id
        self.label = label
        self.centroid = centroid
        self.bbox = bbox
        self.description = description
        self.color= color
        self.material=material
        self.shape=shape
        # GA-186. One entry per SIGHTING: which perception frame, from where, at what
        # centroid. `association.py` has needed this since it was written and the tree
        # recorded it nowhere, which is why co-visibility could not be used as a constraint
        # and why appearance had no way to know whether two views were comparable. Held on
        # the object rather than in a side table so a sighting cannot outlive its subject.
        # Populated by ObjectManagerService._record_sighting; empty is legitimate and every
        # channel that needs observations ABSTAINS on it rather than assuming.
        self.observations = []
