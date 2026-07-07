// 2.13.17 : Spark 4.1.1 est compilé avec et appelle MurmurHash3.caseClassHash
// (absent avant 2.13.17) — une scala-library plus ancienne casse au runtime.
ThisBuild / scalaVersion := "2.13.17"
ThisBuild / version      := "0.1.0"

// Doit matcher la version du spark-submit installé (spark-submit --version)
val sparkVersion       = "4.1.1"
// ≥ 1.20 requis : les modèles exportés par skl2onnx/onnx récents (IR v10+)
// ne se chargent pas avec onnxruntime 1.18. 1.22 : DLL native ne s'initialise
// pas sur cette machine (UnsatisfiedLinkError).
val onnxRuntimeVersion = "1.20.0"

lazy val commonSettings = Seq(
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core"  % sparkVersion % "provided",
    "org.apache.spark" %% "spark-sql"   % sparkVersion % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
    "org.scala-lang.modules" %% "scala-xml" % "2.3.0",
    "org.scalatest" %% "scalatest" % "3.2.19" % Test
  ),
  assembly / assemblyMergeStrategy := {
    case PathList("META-INF", "MANIFEST.MF")          => MergeStrategy.discard
    case PathList("META-INF", "native-image", _*)     => MergeStrategy.discard
    // Les fichiers services/ enregistrent les DataSource et les loaders natifs ONNX :
    // les concaténer au lieu de les jeter, sinon ClassNotFound au runtime.
    case PathList("META-INF", "services", _*)         => MergeStrategy.filterDistinctLines
    case PathList("META-INF", xs @ _*)                => MergeStrategy.discard
    case x if x.endsWith(".proto")                    => MergeStrategy.first
    case x if x.endsWith(".so") || x.endsWith(".dll") => MergeStrategy.first
    case _                                            => MergeStrategy.first
  }
)

lazy val common = (project in file("common"))
  .settings(commonSettings)
  .settings(name := "common")

lazy val ml = (project in file("ml"))
  .dependsOn(common)
  .settings(commonSettings)
  .settings(
    name := "ml",
    libraryDependencies += "com.microsoft.onnxruntime" % "onnxruntime" % onnxRuntimeVersion
  )

lazy val producer = (project in file("producer"))
  .dependsOn(common)
  .settings(commonSettings)
  .settings(
    name := "producer",
    assembly / mainClass := Some("com.project.producer.ProducerApp")
  )

lazy val consumer = (project in file("consumer"))
  .dependsOn(common, ml)
  .settings(commonSettings)
  .settings(
    name := "consumer",
    assembly / mainClass := Some("com.project.consumer.ConsumerApp"),
    libraryDependencies += "com.microsoft.onnxruntime" % "onnxruntime" % onnxRuntimeVersion
  )

lazy val root = (project in file("."))
  .aggregate(common, ml, producer, consumer)
  .settings(name := "spark-batch-streaming")
