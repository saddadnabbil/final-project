"use client";

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import LoginForm from "./LoginForm";
import RegisterForm from "./RegisterForm";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { Languages } from "lucide-react";

export default function NavigationBar() {
  const { user, logout, isAuthenticated, isLoading } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");

  const handleAuthSuccess = () => {
    setShowAuthModal(false);
  };

  const switchToLogin = () => {
    setAuthMode("login");
  };

  const switchToRegister = () => {
    setAuthMode("register");
  };

  return (
    <>
      <nav className="bg-background border-b">
        <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo/Brand */}
            <div className="flex items-center gap-2">
              <Languages className="w-5 h-5 sm:w-6 sm:h-6 text-primary" />
              <h1 className="text-base sm:text-xl font-bold text-primary hidden sm:block">
                Sundanese Translator
              </h1>
            </div>

            {/* Auth Section */}
            <div className="flex items-center gap-2 sm:gap-4">
              {isLoading ? (
                /* Show skeleton while loading */
                <div className="flex items-center gap-2 sm:gap-4">
                  <Skeleton className="h-4 w-20 sm:w-32" />
                  <Skeleton className="h-9 w-16 sm:w-20" />
                </div>
              ) : isAuthenticated ? (
                <>
                  <div className="flex items-center gap-1 sm:gap-2">
                    {/* Hide "Welcome," text on mobile, show only username */}
                    <span className="text-xs sm:text-sm text-foreground">
                      <span className="hidden sm:inline">Welcome, </span>
                      <span className="font-medium">{user?.username}</span>
                    </span>
                    {/* Show badge only on tablet and up */}
                    <Badge variant="secondary" className="hidden md:inline-flex text-xs">
                      {user?.stats?.total_translations || 0} translations
                    </Badge>
                  </div>
                  <Button onClick={logout} variant="outline" size="sm" className="text-xs sm:text-sm h-8 sm:h-9 px-2 sm:px-4">
                    Logout
                  </Button>
                </>
              ) : (
                <Button
                  onClick={() => {
                    setAuthMode("login");
                    setShowAuthModal(true);
                  }}
                  size="sm"
                  className="text-xs sm:text-sm h-8 sm:h-9 px-3 sm:px-4"
                >
                  Login
                </Button>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Auth Dialog */}
      <Dialog open={showAuthModal} onOpenChange={setShowAuthModal}>
        <DialogContent className="sm:max-w-md w-[calc(100%-2rem)] mx-auto">
          <DialogHeader>
            <DialogTitle className="text-lg sm:text-xl">
              {authMode === "login" ? "Login" : "Register"}
            </DialogTitle>
            <DialogDescription className="text-xs sm:text-sm">
              {authMode === "login"
                ? "Sign in to access your translation history"
                : "Create an account to save your translations"}
            </DialogDescription>
          </DialogHeader>
          <div className="mt-3 sm:mt-4">
            {authMode === "login" ? (
              <LoginForm
                onSuccess={handleAuthSuccess}
                onSwitchToRegister={switchToRegister}
              />
            ) : (
              <RegisterForm
                onSuccess={handleAuthSuccess}
                onSwitchToLogin={switchToLogin}
              />
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
