"use client";

import { useState } from "react";
import {
  Upload,
  FileAudio,
  Video,
  Languages,
  History,
  Info,
  Heart,
  X,
  Menu,
  User,
  LogOut,
  Home,
} from "lucide-react";

export default function WireframePage() {
  const [activeTab, setActiveTab] = useState<string>("translate");
  const [hasFile, setHasFile] = useState(false);
  const [fileType, setFileType] = useState<"audio" | "video">("audio");
  const [showResult, setShowResult] = useState(false);
  const [showLoginPopup, setShowLoginPopup] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginTab, setLoginTab] = useState<"login" | "register">("login");

  return (
    <div className="min-h-screen bg-white">
      {/* Header Wireframe */}
      <div className="border-b-2 border-black">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 border-2 border-black rounded"></div>
              <div className="w-32 h-6 border-2 border-black"></div>
            </div>

            {/* User Menu */}
            <div className="flex items-center gap-4">
              <div className="w-24 h-8 border-2 border-black rounded"></div>
              <div className="w-8 h-8 border-2 border-black rounded-full"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tabs */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex border-2 border-black rounded-lg overflow-hidden">
            <button
              onClick={() => setActiveTab("translate")}
              className={`px-6 py-2 border-r-2 border-black flex items-center gap-2 ${
                activeTab === "translate" ? "bg-black text-white" : "bg-white"
              }`}
            >
              <Languages className="w-4 h-4" />
              Translate
            </button>
            <button
              onClick={() => setActiveTab("history")}
              className={`px-6 py-2 flex items-center gap-2 ${
                activeTab === "history" ? "bg-black text-white" : "bg-white"
              }`}
            >
              <History className="w-4 h-4" />
              History
            </button>
          </div>
        </div>

        {/* Translate Tab Content */}
        {activeTab === "translate" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Column */}
            <div className="space-y-6">
              {/* Upload Card */}
              <div className="border-2 border-black">
                {/* Card Header */}
                <div className="border-b-2 border-black p-4">
                  <div className="flex items-center gap-2 mb-1">
                    <Upload className="w-5 h-5" />
                    <div className="w-40 h-5 border-2 border-black"></div>
                  </div>
                  <div className="w-64 h-4 border border-black mt-2"></div>
                </div>

                {/* Card Content */}
                <div className="p-6">
                  {!hasFile ? (
                    <div className="space-y-4">
                      {/* Info Alert */}
                      <div className="border-2 border-black p-4">
                        <div className="flex gap-2">
                          <Info className="w-4 h-4 flex-shrink-0" />
                          <div className="space-y-1 flex-1">
                            <div className="w-full h-3 border border-black"></div>
                            <div className="w-5/6 h-3 border border-black"></div>
                          </div>
                        </div>
                      </div>

                      {/* Upload Area */}
                      <div className="border-2 border-dashed border-black p-12">
                        <div className="flex flex-col items-center gap-4">
                          <div className="w-16 h-16 border-2 border-black rounded"></div>
                          <div className="space-y-2 text-center">
                            <div className="w-48 h-4 border-2 border-black mx-auto"></div>
                            <div className="w-32 h-3 border border-black mx-auto"></div>
                          </div>
                          <div className="w-32 h-10 border-2 border-black rounded"></div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {/* Preview */}
                      {fileType === "video" ? (
                        <div className="border-2 border-black aspect-video flex items-center justify-center">
                          <Video className="w-16 h-16" />
                        </div>
                      ) : (
                        <div className="border-2 border-black p-6">
                          <div className="flex items-center gap-3 mb-4">
                            <FileAudio className="w-8 h-8" />
                            <div className="flex-1 space-y-2">
                              <div className="w-48 h-4 border-2 border-black"></div>
                              <div className="w-24 h-3 border border-black"></div>
                            </div>
                          </div>
                          <div className="w-full h-10 border-2 border-black rounded"></div>
                        </div>
                      )}

                      {/* Buttons */}
                      <button className="w-full h-12 border-2 border-black rounded bg-black text-white flex items-center justify-center gap-2">
                        <Languages className="w-4 h-4" />
                        Process & Translate
                      </button>
                      <button
                        onClick={() => setHasFile(false)}
                        className="w-full h-10 border-2 border-black rounded flex items-center justify-center gap-2"
                      >
                        <X className="w-4 h-4" />
                        Choose Another File
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Instructions Card */}
              <div className="border-2 border-black">
                <div className="border-b-2 border-black p-4">
                  <div className="flex items-center gap-2">
                    <Info className="w-5 h-5" />
                    <div className="w-24 h-5 border-2 border-black"></div>
                  </div>
                </div>
                <div className="p-6">
                  <ol className="space-y-3">
                    {[1, 2, 3, 4].map((num) => (
                      <li key={num} className="flex gap-3">
                        <span className="font-bold">{num}.</span>
                        <div className="flex-1 space-y-1">
                          <div className="w-full h-3 border border-black"></div>
                          <div className="w-4/5 h-3 border border-black"></div>
                        </div>
                      </li>
                    ))}
                  </ol>
                </div>
              </div>
            </div>

            {/* Right Column - Results */}
            <div>
              <div className="border-2 border-black min-h-[500px]">
                {/* Card Header */}
                <div className="border-b-2 border-black p-4">
                  <div className="w-24 h-5 border-2 border-black mb-2"></div>
                  <div className="w-56 h-4 border border-black"></div>
                </div>

                {/* Card Content */}
                <div className="p-6">
                  {!showResult ? (
                    <div className="flex items-center justify-center h-[400px]">
                      <div className="text-center space-y-4">
                        <div className="flex justify-center gap-4">
                          <FileAudio className="w-12 h-12" />
                          <Video className="w-12 h-12" />
                        </div>
                        <div className="space-y-2">
                          <div className="w-48 h-5 border-2 border-black mx-auto"></div>
                          <div className="w-40 h-4 border border-black mx-auto"></div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      {/* Transcription */}
                      <div className="border-2 border-black">
                        <div className="border-b-2 border-black p-3 bg-gray-100">
                          <div className="w-32 h-4 border-2 border-black"></div>
                        </div>
                        <div className="p-4 space-y-2">
                          <div className="w-full h-3 border border-black"></div>
                          <div className="w-11/12 h-3 border border-black"></div>
                          <div className="w-10/12 h-3 border border-black"></div>
                        </div>
                      </div>

                      {/* Translation */}
                      <div className="border-2 border-black">
                        <div className="border-b-2 border-black p-3 bg-gray-100">
                          <div className="w-28 h-4 border-2 border-black"></div>
                        </div>
                        <div className="p-4 space-y-2">
                          <div className="w-full h-3 border border-black"></div>
                          <div className="w-11/12 h-3 border border-black"></div>
                          <div className="w-9/12 h-3 border border-black"></div>
                        </div>
                      </div>

                      {/* Metadata */}
                      <div className="border-2 border-black">
                        <div className="border-b-2 border-black p-3 bg-gray-100">
                          <div className="w-24 h-4 border-2 border-black"></div>
                        </div>
                        <div className="p-4 space-y-3">
                          <div className="flex justify-between items-center">
                            <div className="w-24 h-3 border border-black"></div>
                            <div className="w-32 h-3 border border-black"></div>
                          </div>
                          <div className="flex justify-between items-center">
                            <div className="w-28 h-3 border border-black"></div>
                            <div className="w-24 h-3 border border-black"></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* History Tab Content */}
        {activeTab === "history" && (
          <div className="border-2 border-black">
            <div className="border-b-2 border-black p-4">
              <div className="w-32 h-5 border-2 border-black"></div>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                {[1, 2, 3].map((item) => (
                  <div key={item} className="border-2 border-black p-4">
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 border-2 border-black rounded"></div>
                        <div className="space-y-2">
                          <div className="w-48 h-4 border-2 border-black"></div>
                          <div className="w-32 h-3 border border-black"></div>
                        </div>
                      </div>
                      <div className="w-8 h-8 border-2 border-black rounded"></div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <div className="space-y-2">
                        <div className="w-24 h-3 border border-black"></div>
                        <div className="w-full h-3 border border-black"></div>
                        <div className="w-5/6 h-3 border border-black"></div>
                      </div>
                      <div className="space-y-2">
                        <div className="w-24 h-3 border border-black"></div>
                        <div className="w-full h-3 border border-black"></div>
                        <div className="w-4/6 h-3 border border-black"></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-12 pt-8 border-t-2 border-black">
          <div className="text-center space-y-2">
            <div className="flex items-center justify-center gap-2">
              <div className="w-32 h-4 border-2 border-black"></div>
              <Heart className="w-4 h-4" />
              <div className="w-32 h-4 border-2 border-black"></div>
            </div>
            <div className="w-64 h-3 border border-black mx-auto"></div>
          </div>
        </div>
      </main>

      {/* Login Popup */}
      {showLoginPopup && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white border-4 border-black w-full max-w-md mx-4">
            {/* Popup Header */}
            <div className="border-b-2 border-black p-4 flex items-center justify-between">
              <div className="w-32 h-5 border-2 border-black"></div>
              <button
                onClick={() => setShowLoginPopup(false)}
                className="w-8 h-8 border-2 border-black rounded flex items-center justify-center"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Popup Content */}
            <div className="p-6 space-y-6">
              {/* Tabs */}
              <div className="inline-flex border-2 border-black rounded-lg overflow-hidden w-full">
                <button
                  onClick={() => setLoginTab("login")}
                  className={`flex-1 px-4 py-2 border-r-2 border-black ${
                    loginTab === "login" ? "bg-black text-white" : "bg-white"
                  }`}
                >
                  Login
                </button>
                <button
                  onClick={() => setLoginTab("register")}
                  className={`flex-1 px-4 py-2 ${
                    loginTab === "register" ? "bg-black text-white" : "bg-white"
                  }`}
                >
                  Register
                </button>
              </div>

              {/* Login Form */}
              {loginTab === "login" && (
                <div className="space-y-4">
                  {/* Email/Username */}
                  <div className="space-y-2">
                    <div className="w-24 h-4 border border-black"></div>
                    <div className="w-full h-10 border-2 border-black rounded"></div>
                  </div>

                  {/* Password */}
                  <div className="space-y-2">
                    <div className="w-20 h-4 border border-black"></div>
                    <div className="w-full h-10 border-2 border-black rounded"></div>
                  </div>

                  {/* Remember Me */}
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-black"></div>
                    <div className="w-28 h-3 border border-black"></div>
                  </div>
                </div>
              )}

              {/* Register Form */}
              {loginTab === "register" && (
                <div className="space-y-4">
                  {/* Name */}
                  <div className="space-y-2">
                    <div className="w-16 h-4 border border-black"></div>
                    <div className="w-full h-10 border-2 border-black rounded"></div>
                  </div>

                  {/* Email */}
                  <div className="space-y-2">
                    <div className="w-16 h-4 border border-black"></div>
                    <div className="w-full h-10 border-2 border-black rounded"></div>
                  </div>

                  {/* Password */}
                  <div className="space-y-2">
                    <div className="w-20 h-4 border border-black"></div>
                    <div className="w-full h-10 border-2 border-black rounded"></div>
                  </div>

                  {/* Confirm Password */}
                  <div className="space-y-2">
                    <div className="w-36 h-4 border border-black"></div>
                    <div className="w-full h-10 border-2 border-black rounded"></div>
                  </div>

                  {/* Terms & Conditions */}
                  <div className="flex items-start gap-2">
                    <div className="w-4 h-4 border-2 border-black mt-0.5"></div>
                    <div className="flex-1 space-y-1">
                      <div className="w-full h-3 border border-black"></div>
                      <div className="w-3/4 h-3 border border-black"></div>
                    </div>
                  </div>
                </div>
              )}

              {/* Submit Button */}
              <button
                onClick={() => {
                  setIsLoggedIn(true);
                  setShowLoginPopup(false);
                }}
                className="w-full h-12 border-2 border-black rounded bg-black text-white font-bold"
              >
                {loginTab === "login" ? "LOGIN" : "REGISTER"}
              </button>

              {/* Divider */}
              <div className="flex items-center gap-3">
                <div className="flex-1 h-px border-t border-black"></div>
                <div className="w-8 h-3 border border-black"></div>
                <div className="flex-1 h-px border-t border-black"></div>
              </div>

              {/* Social Login */}
              <div className="space-y-3">
                <button className="w-full h-10 border-2 border-black rounded flex items-center justify-center gap-2">
                  <div className="w-5 h-5 border-2 border-black rounded-full"></div>
                  <div className="w-32 h-4 border-2 border-black"></div>
                </button>
                <button className="w-full h-10 border-2 border-black rounded flex items-center justify-center gap-2">
                  <div className="w-5 h-5 border-2 border-black rounded-full"></div>
                  <div className="w-32 h-4 border-2 border-black"></div>
                </button>
              </div>

              {/* Footer Links */}
              <div className="text-center space-y-2">
                {loginTab === "login" ? (
                  <>
                    <div className="w-40 h-3 border border-black mx-auto"></div>
                    <div className="flex items-center justify-center gap-1">
                      <div className="w-32 h-3 border border-black"></div>
                      <button
                        onClick={() => setLoginTab("register")}
                        className="w-16 h-3 border-2 border-black"
                      ></button>
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center gap-1">
                    <div className="w-36 h-3 border border-black"></div>
                    <button
                      onClick={() => setLoginTab("login")}
                      className="w-16 h-3 border-2 border-black"
                    ></button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Control Panel */}
      <div className="fixed bottom-4 right-4 border-2 border-black bg-white p-4 space-y-2 max-h-[90vh] overflow-y-auto">
        <div className="text-sm font-bold mb-3">Wireframe Controls</div>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={hasFile}
            onChange={(e) => setHasFile(e.target.checked)}
            className="w-4 h-4"
          />
          <span className="text-sm">Show file uploaded</span>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={showResult}
            onChange={(e) => setShowResult(e.target.checked)}
            className="w-4 h-4"
          />
          <span className="text-sm">Show results</span>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={showLoginPopup}
            onChange={(e) => setShowLoginPopup(e.target.checked)}
            className="w-4 h-4"
          />
          <span className="text-sm">Show login popup</span>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={isLoggedIn}
            onChange={(e) => setIsLoggedIn(e.target.checked)}
            className="w-4 h-4"
          />
          <span className="text-sm">Logged in state</span>
        </label>
        <div className="pt-2 border-t-2 border-black">
          <div className="text-xs font-semibold mb-2">File Type:</div>
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="fileType"
              checked={fileType === "audio"}
              onChange={() => setFileType("audio")}
              className="w-4 h-4"
            />
            <span className="text-sm">Audio</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="fileType"
              checked={fileType === "video"}
              onChange={() => setFileType("video")}
              className="w-4 h-4"
            />
            <span className="text-sm">Video</span>
          </label>
        </div>
        <div className="pt-2 border-t-2 border-black">
          <div className="text-xs font-semibold mb-2">Popup Tab:</div>
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="loginTab"
              checked={loginTab === "login"}
              onChange={() => setLoginTab("login")}
              className="w-4 h-4"
            />
            <span className="text-sm">Login</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="loginTab"
              checked={loginTab === "register"}
              onChange={() => setLoginTab("register")}
              className="w-4 h-4"
            />
            <span className="text-sm">Register</span>
          </label>
        </div>
      </div>
    </div>
  );
}
