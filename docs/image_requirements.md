# Image Requirements for ControlNet Training

## By Conditioning Type

### 🖼️ **Canny Edge Conditioning**

**Best Image Types:**
- Clear object boundaries and edges
- Good contrast between subjects and backgrounds
- Architectural scenes (buildings have strong edges)
- Objects with defined shapes (furniture, vehicles, tools)
- Line art and illustrations

**Examples:**
- Buildings and architecture
- Furniture and interior scenes  
- Vehicles (cars, bikes, planes)
- People with clear silhouettes
- Objects against clean backgrounds

**Avoid:**
- Very blurry or soft images
- Images with too much fine detail/texture
- Heavily shadowed scenes
- Abstract patterns without clear edges

### 🏔️ **Depth Map Conditioning**

**Best Image Types:**
- Scenes with clear depth variation
- 3D objects and environments
- Indoor scenes with furniture
- Outdoor landscapes with foreground/background
- Portrait shots with depth of field

**Examples:**
- Room interiors with furniture at different depths
- Landscapes with mountains, trees, and horizon
- Street scenes with buildings and people
- Close-up portraits with background blur
- Still life with objects at various distances

**Avoid:**
- Flat, 2D images (paintings, drawings)
- Images with uniform depth (like walls)
- Very abstract or surreal content
- Macro photography without depth cues

### 🤸 **Human Pose Conditioning**

**Best Image Types:**
- Clear, full-body human figures
- Various poses and activities
- Different ages, genders, ethnicities
- Sports and action poses
- Dance and expressive movements

**Examples:**
- People exercising, dancing, playing sports
- Professional portraits and headshots
- Candid photos of people in action
- Fashion photography with pose variety
- Group photos with multiple people

**Requirements:**
- Clearly visible human figures
- Unobstructed body parts
- Various pose types (standing, sitting, jumping, etc.)
- Different viewpoints (front, side, back, 3/4 view)

### 🎨 **Segmentation Conditioning**

**Best Image Types:**
- Scenes with distinct, separable objects
- Clear boundaries between different elements
- Indoor/outdoor scenes with multiple objects
- Images where you can easily identify different parts

**Examples:**
- Living rooms (sofa, table, lamp, TV, etc.)
- Kitchen scenes (appliances, counters, cabinets)
- Outdoor scenes (sky, trees, grass, people, cars)
- Street photography (buildings, road, sidewalk, signs)

### 🌊 **Normal Map Conditioning**

**Best Image Types:**
- Textured surfaces and materials
- 3D objects with surface detail
- Fabric, stone, wood, metal textures
- Faces and skin (for portrait work)

**Examples:**
- Close-ups of textured materials
- Fabric and clothing details
- Stone, brick, and concrete surfaces
- Human faces and skin textures
- Natural textures (tree bark, rock formations)