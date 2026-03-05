import { useState, useEffect, useRef } from 'react';

const SECTIONS = [
    { id: 'abstract', title: 'Abstract' },
    { id: 'introduction', title: '1. Introduction' },
    { id: 'literature', title: '2. Literature Review' },
    { id: 'methods', title: '3. Materials & Methods' },
    { id: 'results', title: '4. Results & Evaluation' },
    { id: 'conclusion', title: '5. Conclusion' },
    { id: 'references', title: '6. References' },
];

export default function DocsPage() {
    const [activeSection, setActiveSection] = useState('abstract');
    const contentRef = useRef(null);

    useEffect(() => {
        const container = contentRef.current;
        if (!container) return;
        const handleScroll = () => {
            const headings = container.querySelectorAll('[data-section]');
            let current = 'abstract';
            headings.forEach((h) => {
                const rect = h.getBoundingClientRect();
                const containerRect = container.getBoundingClientRect();
                if (rect.top - containerRect.top < 120) {
                    current = h.getAttribute('data-section');
                }
            });
            setActiveSection(current);
        };
        container.addEventListener('scroll', handleScroll);
        return () => container.removeEventListener('scroll', handleScroll);
    }, []);

    const scrollTo = (id) => {
        const el = contentRef.current?.querySelector(`[data-section="${id}"]`);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };

    return (
        <div className="docs-page">
            {/* Table of Contents sidebar */}
            <nav className="docs-toc">
                <div className="docs-toc-header">
                    <span className="material-symbols-outlined">description</span>
                    <span>Table of Contents</span>
                </div>
                <ul className="docs-toc-list">
                    {SECTIONS.map((s) => (
                        <li key={s.id}>
                            <button
                                className={activeSection === s.id ? 'active' : ''}
                                onClick={() => scrollTo(s.id)}
                            >
                                <span className="toc-marker" />
                                {s.title}
                            </button>
                        </li>
                    ))}
                </ul>
                <div className="docs-toc-footer">
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>info</span>
                    <span>Research Paper — G-12</span>
                </div>
            </nav>

            {/* Main document content */}
            <div className="docs-content" ref={contentRef}>
                <div className="docs-content-inner">

                    {/* Title Block */}
                    <div className="docs-title-block">
                        <div className="docs-title-decoration" />
                        <h1>A Procedural Art Generator Using Lindenmayer Systems</h1>
                        <div className="docs-title-rule" />
                    </div>

                    {/* Abstract */}
                    <section data-section="abstract">
                        <h2 className="docs-heading">Abstract</h2>
                        <p className="docs-abstract-text">
                            This research addresses the challenge of generating complex and aesthetically diverse digital art through automated means. Current procedural content generation tools often suffer from a narrow focus, limiting their expressive range by relying on a single generative technique. We propose a novel, integrated procedural art generator designed to overcome these limitations. The methodology centers on a hybrid architecture that combines the structural grammar of stochastic Lindenmayer Systems (L-systems) with the geometric complexity of classic fractal algorithms. Specifically, the system synergistically integrates stochastic 0L-systems for generating varied, organic branching structures with dedicated renderers for Iterated Function Systems (IFS). This fusion allows for the creation of artworks where intricate, self-similar fractal patterns can be used as textures or terminal elements within larger, procedurally defined hierarchical structures. The system is implemented within an accessible, web-based framework, providing artists and designers with high-level parametric control over grammar rules, iteration depth, and geometric interpretation, without requiring programming expertise. The primary contribution of this work is a unified, extensible tool capable of producing a wide variety of novel artistic outputs. This research demonstrates the expressive power that emerges from the synthesis of complementary procedural generation paradigms.
                        </p>
                    </section>

                    <div className="docs-divider" />

                    {/* 1. Introduction */}
                    <section data-section="introduction">
                        <h2 className="docs-heading">1. Introduction</h2>

                        <h3 className="docs-subheading">1.1 Context and Strategic Importance</h3>
                        <p>
                            Procedural content generation (PCG) represents a strategically important paradigm in digital media, offering a powerful alternative to the traditionally high-cost and labor-intensive process of manual content creation. Within this domain, formal, grammar-based systems provide a robust method for encoding complex visual patterns and hierarchical structures into a compact set of rules. These systems, which define a clear separation between generative logic and final output, serve as a foundational technology for automated and semi-automated artistic creation, enabling the rapid generation of a large number of assets with comparable styles. This approach is situated within the broader context of procedural modeling for virtual worlds, where such automation is critical.
                        </p>

                        <h3 className="docs-subheading">1.2 Problem Statement</h3>
                        <p>
                            Despite the power of individual procedural techniques, existing tools for procedural art are often limited by their specialization. Many generators focus on a single methodology, such as Lindenmayer systems for biological growth, shape grammars for architecture, or fractal algorithms for geometric patterns. This specialization restricts the structural and textural variety of the generated art. The literature provides numerous examples of powerful, yet domain-specific, procedural models, such as systems for generating video game levels or 3D creatures. While highly effective in their respective domains, these tools highlight the absence of a general, hybrid system designed specifically for broad artistic exploration.
                        </p>

                        <h3 className="docs-subheading">1.3 Motivation and Research Gap</h3>
                        <p>
                            The motivation for this research stems from the potential for greater expressive power that lies at the intersection of different generative families. By combining the recursive, branching logic of Lindenmayer Systems (L-systems), capable of producing tree-like structures, with the infinite, self-similar geometric detail of classic fractals, it becomes possible to generate a richer and more nuanced class of artistic outputs. The current research landscape reveals a significant gap: there is a lack of an accessible, unified framework that allows artists and designers to leverage multiple, complementary procedural generation techniques simultaneously within a single creative workflow.
                        </p>

                        <h3 className="docs-subheading">1.4 Proposed Solution</h3>
                        <p>
                            To address this gap, we propose a web-based procedural art generator built on the Streamlit framework. The system is designed as a hybrid generative model that integrates three core components: stochastic L-systems and Iterated Function Systems (IFS) models. This top-down approach separates the generative logic from the user interface and provides creators with high-level, semantically meaningful control over the final artistic output. L-systems form the primary structural backbone of a piece, while fractal renderers can be invoked to generate detailed textures or terminal &ldquo;leaf&rdquo; elements, enabling a unique synthesis of organic structure and geometric complexity.
                        </p>

                        <h3 className="docs-subheading">1.5 Contributions</h3>
                        <p>The key contributions of this research are as follows:</p>
                        <ol className="docs-list">
                            <li>
                                <strong>A Novel Hybrid Architecture:</strong> The design of a system that synergistically combines grammar-based rewriting systems (L-systems) with other families of fractal algorithms (IFS), which have traditionally been treated as separate domains.
                            </li>
                            <li>
                                <strong>Integration of Stochastic and Geometric Generation:</strong> The fusion of stochastic L-systems, which introduce controlled randomness to produce organic and varied structures, with deterministic fractal rendering, enabling a unique blend of emergent variation and intricate geometric detail.
                            </li>
                            <li>
                                <strong>An Accessible Implementation:</strong> The development of a user-friendly tool using a modern web framework, making advanced procedural art techniques accessible to a broader audience of non-programmers and addressing a key challenge in the usability of such systems.
                            </li>
                        </ol>
                    </section>

                    <div className="docs-divider" />

                    {/* 2. Literature Review */}
                    <section data-section="literature">
                        <h2 className="docs-heading">2. Literature Review</h2>

                        <h3 className="docs-subheading">2.1 Context and Overview</h3>
                        <p>
                            This section surveys the foundational concepts of procedural modeling that underpin the proposed system. It focuses on the theoretical basis of grammar-based systems, particularly Lindenmayer systems, and examines their relationship to other generative techniques like shape grammars and fractal algorithms. By reviewing the existing body of research, this section establishes the theoretical underpinnings of our proposed hybrid generator and situates it within the broader field of computational creativity.
                        </p>

                        <h3 className="docs-subheading">2.2 Lindenmayer Systems (L-Systems)</h3>
                        <p>
                            Lindenmayer systems, or L-systems, were introduced by Aristid Lindenmayer in 1968 as a formal model for the development and growth of simple organisms. They are a type of parallel string rewriting system where rules are applied simultaneously to all symbols in a string during each derivation step.
                        </p>

                        <h4 className="docs-sub-subheading">Classical 0L-Systems and Turtle Graphics</h4>
                        <p>
                            The simplest form of L-systems are context-free, or 0L-systems, where the rewriting rule for a symbol does not depend on its neighbors. The geometric interpretation of the strings generated by an L-system is most commonly achieved using &ldquo;turtle graphics.&rdquo; A virtual turtle interprets characters in the string as commands:
                        </p>
                        <ul className="docs-list docs-list-code">
                            <li><code>F</code> — Move forward by a set distance and draw a line</li>
                            <li><code>+</code> / <code>-</code> — Turn left or right by a predefined angle</li>
                            <li><code>[</code> — Push the turtle&rsquo;s current state (position, orientation) onto a stack</li>
                            <li><code>]</code> — Pop a state from the stack and restore the turtle to that state</li>
                        </ul>
                        <p>
                            The push (<code>[</code>) and pop (<code>]</code>) commands are crucial for generating the branching structures characteristic of plants and other natural forms.
                        </p>

                        <h4 className="docs-sub-subheading">Stochastic and Parametric L-Systems</h4>
                        <p>
                            To introduce variation and create more organic, less repetitive structures, deterministic L-systems can be extended. Stochastic 0L-systems (SOL-systems) associate a probability distribution with the set of production rules for a given symbol. When rewriting a symbol that has multiple possible successor strings, one is chosen based on these probabilities, introducing controlled randomness into the generative process.
                        </p>
                        <p>
                            Parametric L-systems further extend the formalism by allowing symbols (modules) to have associated parameters. These parameters can be passed to successor modules according to the production rules and can influence the geometric interpretation, for example by controlling branch length or angle.
                        </p>

                        <h3 className="docs-subheading">2.3 Procedural Generation and Shape Grammars</h3>
                        <p>
                            Procedural modeling is the general practice of creating content algorithmically from a set of rules and input parameters, rather than manually. It is a wide-ranging field of application encompassing numerous techniques.
                        </p>

                        <h4 className="docs-sub-subheading">Shape Grammars</h4>
                        <p>
                            Closely related to L-systems are shape grammars, a class of production systems that operate directly on geometric shapes rather than strings of symbols. While L-systems first generate a symbolic string that is later interpreted geometrically, shape grammars define rules that directly replace parts of a shape with new shapes. This approach has been successfully applied to generate architectural models and complex 3D creatures from a &ldquo;creature grammar.&rdquo;
                        </p>

                        <h4 className="docs-sub-subheading">Fractals and Iterated Function Systems (IFS)</h4>
                        <p>
                            Fractals are complex, infinitely detailed patterns characterized by self-similarity across different scales. Many natural processes generate fractal-like structures. While L-systems can generate some types of fractals, other families of algorithms are specialized for this purpose.
                        </p>
                        <p>
                            Iterated Function Systems (IFS) provide another powerful method for constructing fractals by repeatedly applying a set of affine transformations to a set of points, which converge to a final fractal shape. These geometric techniques are complementary to the structural, grammar-based generation of L-systems.
                        </p>

                        <h3 className="docs-subheading">2.4 Comparison with Non-Grammar-Based Systems</h3>
                        <p>
                            Grammar-based procedural modeling can be contrasted with other major PCG paradigms, such as those based on evolutionary computing or constraint satisfaction. Systems using genetic algorithms (GAs) employ search-based techniques to explore a vast design space, using a fitness function to guide the evolution of content towards a desired outcome. Constraint satisfaction methods focus on finding a solution that adheres to a set of strict, predefined constraints.
                        </p>
                        <p>
                            The key difference lies in the approach: grammar-based systems excel at declaratively defining recursive and hierarchical structures, making them intuitive for encoding growth-like patterns. In contrast, evolutionary and constraint-based approaches are fundamentally search and optimization techniques, better suited for problems where the goal is to find an optimal configuration within a possibility space defined by a fitness function or a set of constraints.
                        </p>

                        <h3 className="docs-subheading">2.5 Identified Research Gap</h3>
                        <p>
                            This review of the literature confirms that L-systems, shape grammars, and various fractal algorithms are powerful and well-established techniques for procedural generation. However, research has largely treated them as separate domains, with tools and systems typically specializing in one approach. While L-systems excel at creating hierarchical, branching structures and stochastic grammars provide organic variation, they lack the inherent capacity for the intricate geometric detail found in classic fractals.
                        </p>
                        <p>
                            The identified research gap is therefore the lack of a unified, accessible system designed for artistic exploration that integrates the hierarchical, structural capabilities of stochastic L-systems with the detailed, self-similar geometry of classic fractal rendering methods. The following section will detail the architecture of a system designed to fill this gap.
                        </p>
                    </section>

                    <div className="docs-divider" />

                    {/* 3. Materials and Methods */}
                    <section data-section="methods">
                        <h2 className="docs-heading">3. Materials and Methods</h2>

                        <h3 className="docs-subheading">3.1 Context and Overview</h3>
                        <p>
                            This section provides a detailed technical description of the proposed procedural art generator. It outlines the high-level system overview, the software architecture, the core algorithms for generation and rendering, the underlying mathematical formulations, and key implementation details. This blueprint demonstrates how the theoretical concepts discussed in the literature review are synthesized into a cohesive and functional system.
                        </p>

                        <h3 className="docs-subheading">3.2 System Overview</h3>
                        <p>
                            The proposed system is a Streamlit-based procedural art generator combining stochastic L-systems, and IFS models. These components are integrated to work in concert, allowing for a hybrid generation process. The L-system engine generates the primary structure of an artwork—analogous to the trunk and branches of a tree—by deriving an instruction string from a user-defined grammar. The fractal renderers are then used to provide detailed textures or act as terminal elements in the design, such as the leaves or blossoms on the tree. This entire process is controlled through an interactive, web-based user interface provided by the Streamlit framework, which allows a user to define grammars, adjust parameters in real-time, and view the generated art.
                        </p>

                        <h3 className="docs-subheading">3.3 Architecture</h3>
                        <p>The system&rsquo;s architecture is designed as a sequential pipeline that transforms user input into a final image. The data flows through several distinct modules:</p>

                        <h4 className="docs-sub-subheading">User Input</h4>
                        <p>
                            The process begins in the Streamlit web interface, where the user defines the generative parameters. These include the L-system axiom (the starting string), a set of production rules with associated probabilities, the desired number of derivation iterations (depth), and geometric parameters such as turning angles and step sizes. The user also selects which fractal renderer to use for designated terminal symbols.
                        </p>

                        <h4 className="docs-sub-subheading">L-System Expansion Engine</h4>
                        <p>
                            This core component receives the user-defined grammar and axiom. It performs <em>n</em> derivation steps to expand the axiom into a final, complex instruction string. Adhering to the principles of Stochastic 0L-systems (SOL-systems), the engine uses the probabilities associated with the rules to perform a weighted random selection when a symbol has multiple possible successors.
                        </p>

                        <h4 className="docs-sub-subheading">Interpreters &amp; Renderers</h4>
                        <ul className="docs-list">
                            <li><strong>Turtle Interpreter:</strong> This module processes standard turtle graphics commands (F, +, -, [, ]) from the string. It translates these commands into a series of vector-based lines and state changes, generating the primary structure of the artwork.</li>
                            <li><strong>IFS &amp; Fractal Renderer:</strong> When the turtle interpreter encounters a special terminal symbol designated by the user, this module is invoked. It pauses turtle interpretation and renders a pre-defined IFS-based fractal (e.g., a Barnsley fern) at the turtle&rsquo;s current position, orientation, and scale.</li>
                        </ul>

                        <h4 className="docs-sub-subheading">Output Stage</h4>
                        <ul className="docs-list">
                            <li><strong>SVG/PNG Renderer:</strong> This module composites the vector data from the turtle interpreter and any raster data from the fractal renderer onto a final canvas, which can be displayed in the web UI or exported in standard image formats.</li>
                            <li><strong>Batch Generator &amp; ZIP Export:</strong> This utility allows the user to generate multiple variations of an artwork by iterating over a range of parameters, such as different random seeds. The resulting collection of images can be exported as a single ZIP archive for convenience.</li>
                        </ul>

                        <h3 className="docs-subheading">3.4 Proposed Algorithm and Model</h3>
                        <p>The step-by-step process of generating a piece of art is as follows:</p>
                        <ol className="docs-list">
                            <li><strong>Grammar Parsing and Rule Selection:</strong> The system first parses the user-defined grammar, storing the production rules and their associated weights (probabilities) in a data structure. During each derivation step, the expansion engine iterates through the current string. For each symbol, it identifies the set of applicable production rules and performs a weighted random selection to choose the successor string. This process is repeated for the specified number of iterations.</li>
                            <li><strong>Turtle Interpretation and State Management:</strong> The final, fully derived string is fed to the turtle interpreter. The turtle maintains a state vector consisting of its (x, y) position, orientation angle, and current scale. The <code>[</code> command pushes this entire state onto a stack, while the <code>]</code> command pops the last state from the stack and restores it. This stack-based state management is what enables the creation of complex, non-destructive branching structures.</li>
                            <li><strong>Hybrid Rendering Pipeline:</strong> The rendering process integrates the different generative models into a single pipeline. The turtle interpreter generates the primary vector paths that form the artwork&rsquo;s skeleton. When a pre-defined fractal-generating terminal symbol is encountered in the instruction string, the renderer pauses the turtle interpretation, generates the specified fractal as either a raster or vector object at the current turtle state, composites it onto the canvas, and then resumes turtle interpretation from the same state.</li>
                            <li><strong>Raster and Vector Conversion:</strong> The final artwork is a composition of vector paths generated by the turtle and potentially raster images from the fractal renderer. This composite canvas is then converted into the desired output format. For PNG export, the canvas is rendered into a pixel-based image. For SVG export, the turtle&rsquo;s vector paths are written as SVG path elements, and any rasterized fractals are either embedded as base64-encoded images or approximated with vector paths.</li>
                        </ol>

                        <h3 className="docs-subheading">3.5 Mathematical Formulation</h3>
                        <p>The system is grounded in the following mathematical concepts:</p>

                        <h4 className="docs-sub-subheading">L-System Rewriting</h4>
                        <p>A stochastic L-system is formally defined as a 4-tuple <strong>S = (V, &omega;, P, &rho;)</strong>, where:</p>
                        <ul className="docs-list">
                            <li><strong>V</strong> is the alphabet (the set of symbols)</li>
                            <li><strong>&omega;</strong> is the axiom, or starting string</li>
                            <li><strong>P</strong> is the set of production rules</li>
                            <li><strong>&rho;</strong> is a probability function that assigns a probability to each rule in P, such that for any symbol A &isin; V, the sum of probabilities of all rules with A as the predecessor is equal to 1</li>
                        </ul>

                        <h4 className="docs-sub-subheading">Probabilistic Rule Selection</h4>
                        <p>
                            For a given symbol A with <em>k</em> possible successor rules, A &rarr; &alpha;&#8321;, A &rarr; &alpha;&#8322;, &hellip;, A &rarr; &alpha;&#8342;, associated with probabilities p&#8321;, p&#8322;, &hellip;, p&#8342; where &Sigma;p&#7522; = 1, the expansion engine selects rule <em>i</em> to rewrite A with probability p&#7522;.
                        </p>

                        <h4 className="docs-sub-subheading">Fractal Generation</h4>
                        <ul className="docs-list">
                            <li><strong>Iterated Function Systems (IFS):</strong> An IFS is defined by a set of affine transformations (e.g., scaling, rotation, translation). A fractal is generated by starting with an initial set of points and repeatedly applying these transformations, with the collection of points converging to the final fractal shape.</li>
                        </ul>

                        <h3 className="docs-subheading">3.6 Implementation Details</h3>
                        <p>The software implementation of the procedural generator is based on the following technologies:</p>
                        <ul className="docs-list">
                            <li><strong>Language:</strong> Python</li>
                            <li><strong>Framework:</strong> Streamlit is used to create the interactive, web-based graphical user interface</li>
                            <li><strong>Core Libraries:</strong>
                                <ul className="docs-list">
                                    <li><em>NumPy</em> — For efficient numerical operations, particularly during the calculation-intensive process of rendering</li>
                                    <li><em>Pillow (PIL)</em> — For creating, manipulating, and saving raster images, used for PNG output</li>
                                    <li><em>JSON</em> — For importing and exporting L-system grammar definitions, allowing users to save and share their creations</li>
                                </ul>
                            </li>
                        </ul>

                        <h4 className="docs-sub-subheading">Key User-Controlled Parameters</h4>
                        <p>The Streamlit interface exposes several key parameters for user control, including:</p>
                        <ul className="docs-list">
                            <li>L-system iteration depth</li>
                            <li>Turtle&rsquo;s turning angle and step size</li>
                            <li>Random seeds for controlling the output of the stochastic processes</li>
                            <li>Specific parameters for the fractal renderers (e.g., the <em>c</em> value for a Julia set)</li>
                        </ul>
                    </section>

                    <div className="docs-divider" />

                    {/* 4. Results and Evaluation */}
                    <section data-section="results">
                        <h2 className="docs-heading">4. Results and Evaluation</h2>

                        <h3 className="docs-subheading">4.1 Context and Overview</h3>
                        <p>
                            This section evaluates the capabilities and performance of the proposed procedural art generator. The evaluation aims to demonstrate the system&rsquo;s expressive range and quantify its performance characteristics. The experimental setup is described, followed by a demonstration of outputs generated by different combinations of the system&rsquo;s modules. Finally, the system&rsquo;s expressive power is qualitatively compared against existing, non-integrated approaches to highlight the benefits of the hybrid architecture.
                        </p>

                        <h3 className="docs-subheading">4.2 Experiment Setup</h3>
                        <p>The performance of the generator was tested in the following environment:</p>
                        <ul className="docs-list">
                            <li><strong>Hardware:</strong> All tests were conducted on a standard consumer-grade computer, reflecting a typical user environment. The machine was an Intel Core i7 processor with 16 GB of RAM.</li>
                            <li><strong>Software:</strong> The system was run on a Windows 11 operating system with Python 3.10. The key libraries included Streamlit 1.15.0, NumPy 1.23.5, and Pillow 9.3.0.</li>
                        </ul>

                        <h3 className="docs-subheading">4.3 Output Demonstration</h3>
                        <p>The following table showcases a variety of outputs generated by the system, along with performance metrics. The generation time measures the duration from parameter submission to the final image rendering.</p>

                        <div className="docs-table-wrapper">
                            <table className="docs-table">
                                <thead>
                                    <tr>
                                        <th>Model Type</th>
                                        <th>Iterations</th>
                                        <th>Gen. Time (s)</th>
                                        <th>Output Size (KB)</th>
                                        <th>Notes</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Simple L-System</td>
                                        <td>5</td>
                                        <td>0.2</td>
                                        <td>15 (SVG)</td>
                                        <td>Deterministic plant-like structure</td>
                                    </tr>
                                    <tr>
                                        <td>Stochastic L-System</td>
                                        <td>7</td>
                                        <td>0.8</td>
                                        <td>45 (SVG)</td>
                                        <td>Organic, asymmetric tree with random branching</td>
                                    </tr>
                                    <tr>
                                        <td>Julia Set</td>
                                        <td>200</td>
                                        <td>1.5</td>
                                        <td>150 (PNG)</td>
                                        <td>Rendered as a terminal element on a simple L-system</td>
                                    </tr>
                                    <tr>
                                        <td>Hybrid (Stochastic L-System + IFS)</td>
                                        <td>6</td>
                                        <td>2.1</td>
                                        <td>210 (PNG)</td>
                                        <td>A stochastic tree structure where each leaf is replaced by an IFS fern</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>

                        <h3 className="docs-subheading">4.4 Comparison with Existing Work</h3>
                        <p>
                            The primary advantage of the proposed hybrid system lies in its superior expressive power and variety compared to tools based on a single generative technique. Furthermore, it retains the declarative, hierarchical control that is a key strength of grammar-based systems. While non-grammar-based approaches, such as evolutionary systems, are powerful for search and optimization problems, they do not offer the same intuitive, rule-based definition of recursive structures. Our hybrid system enhances this grammatical control by integrating complementary geometric techniques.
                        </p>

                        <h4 className="docs-sub-subheading">Expressiveness</h4>
                        <p>
                            The system&rsquo;s expressiveness surpasses that of tools focused on a single method. Classic fractal generators, for instance, can produce images of immense geometric detail and complexity, but they lack the capacity to generate the overarching hierarchical and branching structures that are the hallmark of L-systems. Our system combines these capabilities, allowing for the generation of artworks that possess both global structure and local intricacy.
                        </p>

                        <h4 className="docs-sub-subheading">Variety</h4>
                        <p>
                            When compared to deterministic L-system tools, the integration of stochastic rules (SOL-systems) is a significant advantage. It allows a single grammar to produce an entire family of related but non-identical artworks, avoiding the mechanical and repetitive feel of purely deterministic output. This controlled randomness is essential for creating organic and natural-looking forms.
                        </p>

                        <h4 className="docs-sub-subheading">Integration</h4>
                        <p>
                            The key innovation is the seamless integration of these methods within a single, accessible tool. This unlocks novel artistic possibilities—such as fractal patterns that grow along procedurally generated curves or entire IFS-generated structures serving as the &ldquo;leaves&rdquo; of a stochastic tree—that would be cumbersome or difficult to achieve by manually composing outputs from separate, specialized software.
                        </p>
                    </section>

                    <div className="docs-divider" />

                    {/* 5. Conclusion */}
                    <section data-section="conclusion">
                        <h2 className="docs-heading">5. Conclusion</h2>
                        <p>
                            This research has detailed the design and implementation of a novel procedural art generator that successfully integrates stochastic Lindenmayer Systems with classic fractal rendering techniques, including Iterated Function Systems. The research demonstrates that this hybrid approach enables the creation of a diverse range of complex and aesthetically rich imagery that would be difficult to produce with systems reliant on a single generative paradigm.
                        </p>
                        <p>
                            By combining the structural, hierarchical power of grammar-based systems with the intricate geometric detail of fractals, the system opens up new avenues for artistic exploration. Future work could extend this framework in several promising directions, most notably:
                        </p>
                        <ol className="docs-list">
                            <li><strong>3D Structure Generation:</strong> The expansion of the rendering pipeline to support the generation of three-dimensional structures</li>
                            <li><strong>Advanced Parametric L-Systems:</strong> The implementation of more sophisticated parametric L-systems, which would allow environmental factors within the canvas to dynamically influence the growth and development of the generated forms</li>
                            <li><strong>Interactive Feedback:</strong> Integration of real-time user feedback mechanisms to guide the generative process</li>
                            <li><strong>Machine Learning Integration:</strong> Incorporation of learned models to discover novel grammar rules and fractal parameters</li>
                        </ol>
                    </section>

                    <div className="docs-divider" />

                    {/* 6. References */}
                    <section data-section="references">
                        <h2 className="docs-heading">6. References</h2>
                        <ul className="docs-references">
                            <li>Alfadalat, M. A., Al-Azhari, W., &amp; Dabbour, L. (2023). Procedural Modeling Based Shape Grammar as a Key to Generating Digital Architectural Heritage. <em>ACM Journal on Computing and Cultural Heritage</em>, 16(4), Article 68.</li>
                            <li>Eichhorst, P., &amp; Savitch, W. J. (1980). Growth Functions of Stochastic Lindenmayer Systems. <em>Information and Control</em>, 45(3), 217–228.</li>
                            <li>Guo, J., Jiang, H., Benes, B., Deussen, O., Zhang, X., Lischinski, D., &amp; Huang, H. (2020). Inverse Procedural Modeling of Branching Structures by Inferring L-Systems. <em>ACM Transactions on Graphics</em>, 39(5), Article 155.</li>
                            <li>Guo, X., Shen, H., Wang, G., &amp; Li, R. (2014). Creature grammar for creative modeling of 3D monsters. <em>Graphical Models</em>, 76(5), 376–389.</li>
                            <li>Lindenmayer, A. (1968). Mathematical models for cellular interaction in development. <em>Journal of Theoretical Biology</em>, 18(3), 280–315.</li>
                            <li>Perttunen, J., &amp; Sievänen, R. (2005). Incorporating Lindenmayer systems for architectural development in a functional-structural tree model. <em>Ecological Modelling</em>, 181(4), 479–491.</li>
                            <li>Ryzhikova, Y. V., &amp; Ryzhikov, S. B. (2023). Pattern Analysis of Fractal-Like Systems in the Vicinity of the Critical Point. <em>Moscow University Physics Bulletin</em>, 78(4), 518–521.</li>
                            <li>Sorenson, N., Pasquier, P., &amp; DiPaola, S. (2011). A Generic Approach to Challenge Modeling for the Procedural Creation of Video Game Levels. <em>IEEE Transactions on Computational Intelligence and AI in Games</em>, 3(3), 229–244.</li>
                            <li>Šťava, O., Beneš, B., Měch, R., Aliaga, D. G., &amp; Krištof, P. (2010). Inverse Procedural Modeling by Automatic Generation of L-systems. <em>Computer Graphics Forum</em>, 29(2).</li>
                        </ul>
                    </section>

                    {/* Bottom spacing */}
                    <div style={{ height: 80 }} />

                </div>
            </div>
        </div>
    );
}
