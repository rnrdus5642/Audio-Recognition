using System;
using System.IO;
using System.Linq;
using DomiCube.PhonemeMatching.Unity;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Editor
{
    public static class AudioAnswerBuildSetup
    {
        public const string BootstrapTemplatePath = "Packages/com.domicube.phoneme-matching/Runtime/Unity/Startup/AudioModelStartup.unity";
        public const string BootstrapScenePath = "Assets/AudioAnswerStartup/AudioModelStartup.unity";
        public const string LaunchConfigPath = "Assets/Resources/CRC/AudioAnswerLaunch.json";

        public static void Configure()
        {
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(BootstrapTemplatePath) == null)
                throw new InvalidOperationException("오디오 시작 씬을 찾을 수 없습니다. 패키지를 다시 설치하세요.");
            // Unity 2022 cannot open a scene inside a read-only UPM package for a Player build.
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(BootstrapScenePath) == null)
            {
                if (!AssetDatabase.IsValidFolder("Assets/AudioAnswerStartup"))
                    AssetDatabase.CreateFolder("Assets", "AudioAnswerStartup");
                if (!AssetDatabase.CopyAsset(BootstrapTemplatePath, BootstrapScenePath))
                    throw new InvalidOperationException("오디오 시작 씬을 프로젝트에 만들지 못했습니다.");
            }
            var scenes = EditorBuildSettings.scenes.Where(s => s.path != BootstrapScenePath && s.path != BootstrapTemplatePath).ToArray();
            var game = scenes.FirstOrDefault(s => s.enabled);
            if (game == null)
                throw new InvalidOperationException("Build Settings에 실행할 게임 씬을 추가하고 초기 세팅을 다시 실행하세요.");
            string contents = JsonUtility.ToJson(new AudioAnswerLaunchConfiguration { gameScene = game.path }, true) + "\n";
            if (!File.Exists(LaunchConfigPath) || File.ReadAllText(LaunchConfigPath) != contents)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(LaunchConfigPath));
                File.WriteAllText(LaunchConfigPath, contents, new System.Text.UTF8Encoding(false));
                AssetDatabase.ImportAsset(LaunchConfigPath);
            }
            var desired = new[] { new EditorBuildSettingsScene(BootstrapScenePath, true) }.Concat(scenes).ToArray();
            if (!EditorBuildSettings.scenes.Select(s => (s.path, s.enabled)).SequenceEqual(desired.Select(s => (s.path, s.enabled))))
                EditorBuildSettings.scenes = desired;
        }

        public static void Validate()
        {
            var scenes = EditorBuildSettings.scenes.Where(s => s.enabled).ToArray();
            if (scenes.Length < 2 || scenes[0].path != BootstrapScenePath || !File.Exists(LaunchConfigPath))
                throw new BuildFailedException("오디오 초기 세팅을 실행해 빌드 앱의 시작 씬을 구성하세요.");
            var config = JsonUtility.FromJson<AudioAnswerLaunchConfiguration>(File.ReadAllText(LaunchConfigPath));
            if (config == null || config.gameScene == BootstrapScenePath || !scenes.Any(s => s.path == config.gameScene))
                throw new BuildFailedException("오디오 게임 시작 씬 설정이 Build Settings와 다릅니다. 초기 세팅을 다시 실행하세요.");
        }

        public static void ValidateBuildOptions(BuildPlayerOptions options)
        {
            if (options.target != BuildTarget.StandaloneWindows64 && options.target != BuildTarget.StandaloneWindows)
                throw new BuildFailedException("현재 공용 음성 모델 자동 설치는 Windows PC 빌드용입니다.");
            Validate();
            var config = JsonUtility.FromJson<AudioAnswerLaunchConfiguration>(File.ReadAllText(LaunchConfigPath));
            if (options.scenes == null || options.scenes.Length < 2 || options.scenes[0] != BootstrapScenePath
                || !options.scenes.Contains(config.gameScene))
                throw new BuildFailedException("실제 빌드 씬 목록에 오디오 시작 씬을 첫 항목으로, 게임 씬을 다음 항목으로 포함하세요.");
        }
    }

    // Fail explicitly if a custom build script omits the bootstrap scene.
    public sealed class AudioAnswerBuildGuard : BuildPlayerProcessor
    {
        public override int callbackOrder => 0;
        public override void PrepareForBuild(BuildPlayerContext context)
        {
            if (Resources.Load<TextAsset>("CRC/AudioAnswerProject") == null) return;
            // Inspect actual build options, independent of incremental scene-cache callbacks.
            AudioAnswerBuildSetup.ValidateBuildOptions(context.BuildPlayerOptions);
        }
    }
}
